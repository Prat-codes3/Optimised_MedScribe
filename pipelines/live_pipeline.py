"""
Optimised MedScribe — Live Pipeline (Real-Time Mic Transcription)

This is the CONDUCTOR of the real-time transcription workflow.
It ties together:
    1. sounddevice           — Captures microphone audio chunks.
    2. queue.Queue           — Safely passes audio chunks from the mic thread to the processing thread.
    3. SileroVAD             — Gates incoming audio so we ignore background noise/silence.
    4. WhisperModel (small)  — Transcribes speech chunks with low latency.
    5. LocalAgreementBuffer  — Removes duplicate words between overlapping windows.

THE PRODUCER-CONSUMER PATTERN:
    Audio capture must be extremely fast and never block.  If the computer is busy
    doing a heavy Whisper inference, the microphone buffer might overflow, leading
    to lost audio (glitches/clicks).

    To prevent this, we split the tasks:
    1. The PRODUCER (Mic Thread):
       - A lightweight callback function triggered by sounddevice every 32ms (512 samples).
       - It does zero math. It simply copies the raw audio and pushes it to a Queue.
    2. The CONSUMER (Main Thread):
       - Reads frames from the Queue.
       - Runs VAD on each frame.
       - Accumulates speech frames into a buffer.
       - When the VAD detects an "endpoint" (the speaker paused for >500ms),
         it sends the buffer to Whisper, de-duplicates with LocalAgreement,
         and prints/saves the text.

BENEFITS OVER v1.0:
    - Dedicated VAD gating: Never wastes CPU transcribing silence or breathing.
    - Word-level de-duplication: No repeating phrases.
    - Decoupled from CLI: Can be imported directly into the FastAPI server for WebSockets.
"""

import os
import queue
import time
import sys
from datetime import datetime
from typing import Iterator

import numpy as np

from config import AppConfig
from model_manager import ModelManager
from core.vad import SileroVAD, FRAME_SIZE, SAMPLE_RATE
from core.agreement import LocalAgreementBuffer


class LivePipeline:
    """Orchestrates real-time microphone audio transcription.

    This class captures audio from the system's default microphone,
    filters out silence using Silero VAD, transcribes speaking segments
    using Whisper, and de-duplicates the output using a Local Agreement buffer.

    Attributes:
        config: The AppConfig instance containing live parameters.
        manager: The ModelManager singleton.
        transcriber: The transcriber instance (lazy-loaded).
        vad: The SileroVAD instance.
        agreement_buffer: The LocalAgreementBuffer instance.
        audio_queue: Thread-safe queue to store raw microphone frames.
        is_running: Boolean flag indicating if recording is active.
    """

    def __init__(self, config: AppConfig):
        """Initialize the live pipeline.

        Args:
            config: The centralized AppConfig instance.
        """
        self.config = config
        self.manager = ModelManager(config)

        # Core components
        self.vad = SileroVAD(threshold=self.config.vad_threshold)
        self.agreement_buffer = LocalAgreementBuffer()

        # Thread-safe queue for microphone audio chunks
        self.audio_queue = queue.Queue()
        self.is_running = False

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback triggered by sounddevice whenever new mic audio is ready.

        This runs on a dedicated high-priority audio thread.  It MUST be
        extremely lightweight to avoid audio dropouts.  Its only job is to
        copy the audio data and put it in the queue.

        Args:
            indata: Numpy array of shape (frames, channels) containing mic audio.
            frames: Number of samples in this block.
            time_info: Dict containing stream timing info.
            status: Callback status flags (e.g., input overflow).
        """
        if status:
            print(f"Warning: Audio callback status - {status}", file=sys.stderr)

        # Copy the single-channel audio to prevent overwriting
        chunk = indata[:, 0].copy()

        # Push to queue. If queue is full (which shouldn't happen), drop the oldest
        # chunk to make room for real-time continuity.
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.audio_queue.put_nowait(chunk)

    def start_stream(self, output_file: str | None = None) -> Iterator[str]:
        """Start capturing and transcribing microphone input.

        This method is a generator. It yields newly transcribed words
        in real-time as they are spoken.

        Args:
            output_file: Optional file path to write the session transcript.

        Yields:
            Newly transcribed phrases/words as strings.
        """
        import sounddevice as sd

        # ── Step 1: Initialize/Reset state ────────────────────────────
        self.audio_queue.queue.clear()
        self.vad.reset()
        self.agreement_buffer.reset()
        self.is_running = True

        # Resolve output path
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.abspath(f"live_session_{timestamp}.txt")
        else:
            output_file = os.path.abspath(output_file)

        # ── Step 2: Load Model ────────────────────────────────────────
        # Live mode uses a smaller model (e.g. "small") for low latency
        transcriber = self.manager.create_transcriber()
        print(f"\n--- Live Transcription Starting ---")
        print(f"    Model:   {self.config.live_model}")
        print(f"    Device:  {self.config.device.upper()}")
        print(f"    Saving:  {output_file}")
        print(f"    Speak now... (Press Ctrl+C in terminal to stop)")
        print(f"{'-'*50}\n")

        # Open output file and write header
        with open(output_file, "w", encoding="utf-8") as writer:
            writer.write(
                "==================================================\n"
                "  OPTIMISED MEDSCRIBE LIVE SESSION\n"
                "==================================================\n"
                f"  Started On : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Model      : {self.config.live_model}\n"
                f"  Language   : {self.config.language}\n"
                "==================================================\n\n"
            )
            writer.flush()

            # ── Step 3: Start Input Stream ────────────────────────────
            # We capture audio in chunks matching the VAD FRAME_SIZE (512 samples)
            # 512 samples at 16kHz = 32ms chunks.
            # This provides the lowest possible latency for VAD analysis.
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self._audio_callback,
                blocksize=FRAME_SIZE,
                dtype="float32",
            )

            # Keep track of active speech audio samples
            speech_buffer = []

            # Track session start for absolute timestamps
            session_start = time.time()

            with stream:
                try:
                    while self.is_running:
                        # Get a 32ms audio frame from the queue (blocks if empty)
                        try:
                            frame = self.audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                        # Process this frame through VAD
                        # min_silence_ms=500 is used for detecting sentence endings
                        vad_result = self.vad.process_frame(frame)

                        if vad_result["is_speaking"]:
                            # Speaker is active — accumulate speech samples
                            speech_buffer.append(frame)

                        # Check if speaker has paused long enough to trigger an endpoint
                        endpoint_detected = self.vad.check_endpoint(min_silence_ms=500)

                        # Transcribe if we have speech accumulated and:
                        # 1. An endpoint (pause) was detected, OR
                        # 2. The speech buffer is getting too long (safety limit: 15 seconds)
                        #    to prevent Whisper from running out of context memory.
                        buffer_duration = len(speech_buffer) * (FRAME_SIZE / SAMPLE_RATE)
                        if speech_buffer and (endpoint_detected or buffer_duration >= 15.0):
                            # Concatenate list of frames into a single audio array
                            audio_chunk = np.concatenate(speech_buffer)

                            # Run Whisper inference (beam_size=1, greedy search for speed)
                            raw_text = transcriber.transcribe_chunk(audio_chunk)

                            # Pass text through agreement buffer to remove duplicates
                            new_text = self.agreement_buffer.process(raw_text)

                            if new_text:
                                elapsed_seconds = time.time() - session_start
                                minutes, seconds = divmod(int(elapsed_seconds), 60)
                                time_tag = f"[{minutes:02d}:{seconds:02d}]"

                                # Format text line
                                line = f"{time_tag} {new_text}"

                                # Write to file
                                writer.write(f"{line}\n")
                                writer.flush()

                                # Yield to generator caller (e.g., CLI or WebSocket)
                                yield line

                            # If an endpoint was detected, clear the buffer for the next sentence.
                            # If we hit the 15s safety limit but the speaker is still talking,
                            # we keep the last 1.5 seconds of audio in the buffer to preserve
                            # sliding context for Whisper.
                            if endpoint_detected:
                                speech_buffer = []
                                # Clear agreement history at sentence boundaries so the buffer
                                # doesn't grow indefinitely.
                                self.agreement_buffer.reset()
                            else:
                                # Sliding context (keep last 1.5s of samples)
                                overlap_samples = int(SAMPLE_RATE * 1.5)
                                speech_buffer = [audio_chunk[-overlap_samples:]]

                except KeyboardInterrupt:
                    print("\n\n--- Session ended by user request ---")
                finally:
                    self.is_running = False
                    # Write closing separator to session file
                    writer.write(
                        "\n==================================================\n"
                        f"  End Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        "==================================================\n"
                    )
                    writer.flush()
                    print(f"--- Session saved successfully to: {output_file} ---")

    def stop_stream(self):
        """Stop the active recording session."""
        self.is_running = False
