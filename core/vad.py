"""
Optimised MedScribe — Voice Activity Detection (Silero VAD)

This module answers the question: "Is someone SPEAKING right now?"

In live transcription, microphone audio streams in continuously — but
most of that audio is silence, keyboard typing, breathing, or background
noise (AC, fan, traffic).  If we send ALL of that to Whisper, two bad
things happen:
    1. Whisper "hallucinates" — it generates fake text for silent audio
       (e.g., "Thank you for watching", repeated sentences, random words).
    2. We waste GPU/CPU cycles processing audio that has no speech.

Silero VAD is a tiny neural network (~2 MB) that can classify each small
audio frame as "speech" or "not speech" in under 1 millisecond.  We use
it as a GATEKEEPER — only audio frames marked as speech are accumulated
and eventually sent to Whisper for transcription.

HOW SILERO VAD WORKS (simplified):
    1. You feed it a small audio frame (512 samples at 16kHz = 32ms).
    2. It returns a confidence score between 0.0 and 1.0:
       - 0.0 = definitely silence
       - 0.5 = uncertain
       - 1.0 = definitely speech
    3. You compare the score against a threshold (default 0.5).
       Above threshold → speech.  Below → silence.

    The model is STATEFUL — it remembers what it heard in previous frames,
    so it can handle transitions smoothly (e.g., it won't flip-flop
    between speech and silence mid-word).

ENDPOINT DETECTION:
    Besides filtering silence, the VAD also detects "endpoints" — the
    moment a speaker finishes a sentence or pauses.  This is crucial
    for knowing WHEN to send accumulated audio to Whisper.

    We detect endpoints by tracking how many consecutive frames have
    been silent.  If the speaker has been quiet for longer than a
    threshold (e.g., 500ms), we consider that an endpoint.

    Why not use a fixed timer?
        Because a fixed timer (e.g., "send to Whisper every 3 seconds")
        often cuts words in half.  VAD-based endpoints wait for natural
        pauses in speech, resulting in cleaner sentence boundaries.
"""

import numpy as np
import torch


# ─── Constants ────────────────────────────────────────────────────────
# Silero VAD expects audio at 16kHz sample rate.
SAMPLE_RATE = 16000

# The number of audio samples per frame that Silero VAD processes.
# 512 samples at 16kHz = 32 milliseconds per frame.
# This is the smallest unit of audio the VAD analyzes at a time.
FRAME_SIZE = 512


class SileroVAD:
    """Lightweight Voice Activity Detection using Silero VAD.

    This class wraps Silero VAD behind a clean API.  It processes audio
    frames one at a time and tracks speech/silence state to detect
    when a speaker starts talking, stops talking, and endpoints.

    Attributes:
        threshold: Confidence score above which a frame is "speech".
        model: The loaded Silero VAD neural network.
        _is_speaking: Whether the VAD currently thinks someone is talking.
        _silence_frames: Count of consecutive silent frames (for endpoint).
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize the VAD by loading the Silero model.

        The Silero VAD model is tiny (~2 MB) and loads in under 1 second.
        It is downloaded from torch.hub on first use, then cached locally
        by PyTorch for subsequent runs.

        Args:
            threshold: Confidence score threshold (0.0 to 1.0).
                       Frames scoring above this are classified as speech.
                       - Lower (0.3): more sensitive, catches quiet speech
                         but may also trigger on background noise.
                       - Higher (0.7): less sensitive, only triggers on
                         clear speech but may miss whispers.
                       - Default (0.5): balanced for typical indoor rooms.
        """
        self.threshold = threshold

        # Load the Silero VAD model from torch.hub
        # 'snakers4/silero-vad' is the official repository
        # force_reload=False uses the cached model after first download
        print("--- Loading Silero VAD ---")
        self.model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        print("--- Silero VAD loaded ---")

        # ── Internal State ────────────────────────────────────────────
        # The VAD tracks whether we are currently in a "speaking" or
        # "silence" state.  This avoids reacting to every single frame
        # independently (which would cause rapid toggling).

        self._is_speaking = False
        self._silence_frames = 0

    # ─── Public API ───────────────────────────────────────────────────

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Check if a single audio frame contains speech.

        This is the simplest API — give it a frame, get back True/False.
        Use this when you just need a quick yes/no per frame.

        Args:
            audio_frame: A 1-D numpy array of float32 audio samples.
                         Should be exactly FRAME_SIZE (512) samples.
                         Must be at 16kHz sample rate.

        Returns:
            True if the frame contains speech, False otherwise.
        """
        score = self.get_confidence(audio_frame)
        return score >= self.threshold

    def get_confidence(self, audio_frame: np.ndarray) -> float:
        """Get the speech confidence score for an audio frame.

        Use this when you need the raw score for custom logic
        (e.g., adaptive thresholding, logging, visualization).

        Args:
            audio_frame: A 1-D numpy array of float32 audio samples.

        Returns:
            A float between 0.0 (silence) and 1.0 (speech).
        """
        # Convert numpy array to PyTorch tensor (Silero expects a tensor)
        tensor = torch.from_numpy(audio_frame).float()

        # Run the model — this returns a single confidence score
        # The model is stateful, so it remembers previous frames
        with torch.no_grad():
            score = self.model(tensor, SAMPLE_RATE).item()

        return score

    def process_frame(self, audio_frame: np.ndarray) -> dict:
        """Process a frame and return detailed state information.

        This is the FULL API used by the live pipeline.  It not only
        checks if the frame is speech, but also tracks state transitions
        and detects endpoints.

        State Transitions:
            SILENCE → SPEECH:  "speech_start" event (speaker started talking)
            SPEECH  → SPEECH:  No event (speaker is still talking)
            SPEECH  → SILENCE: Increments silence counter
            SILENCE → SILENCE: Increments silence counter

        Args:
            audio_frame: A 1-D numpy array of float32 audio samples.

        Returns:
            A dictionary with:
            {
                "is_speech": bool,         # Is this frame speech?
                "confidence": float,       # Raw score 0.0-1.0
                "is_speaking": bool,       # Are we in a speaking state?
                "speech_start": bool,      # Did speech just START this frame?
                "silence_frames": int,     # Consecutive silent frames
                "silence_duration_ms": float,  # Silence duration in ms
            }
        """
        score = self.get_confidence(audio_frame)
        frame_is_speech = score >= self.threshold

        speech_start = False

        if frame_is_speech:
            # Reset silence counter — someone is talking
            self._silence_frames = 0

            if not self._is_speaking:
                # Transition: SILENCE → SPEECH
                self._is_speaking = True
                speech_start = True
        else:
            # Increment silence counter
            self._silence_frames += 1

            # We don't immediately set _is_speaking to False here.
            # That's handled by check_endpoint(), because brief pauses
            # between words (50-100ms) are normal and shouldn't end
            # the speaking state.

        # Calculate silence duration in milliseconds
        # Each frame = FRAME_SIZE / SAMPLE_RATE seconds
        frame_duration_ms = (FRAME_SIZE / SAMPLE_RATE) * 1000
        silence_ms = self._silence_frames * frame_duration_ms

        return {
            "is_speech": frame_is_speech,
            "confidence": round(score, 4),
            "is_speaking": self._is_speaking,
            "speech_start": speech_start,
            "silence_frames": self._silence_frames,
            "silence_duration_ms": round(silence_ms, 1),
        }

    def check_endpoint(self, min_silence_ms: float = 500.0) -> bool:
        """Check if the speaker has stopped talking (endpoint detected).

        An "endpoint" means the speaker has been silent for at least
        min_silence_ms milliseconds.  When this happens, it's a good
        time to send accumulated audio to Whisper for transcription.

        Args:
            min_silence_ms: Minimum silence duration (in milliseconds)
                            to consider as an endpoint.
                            - 300ms: aggressive, cuts quickly (may split words)
                            - 500ms: balanced (default — good for conversation)
                            - 1000ms: conservative (waits for long pauses)

        Returns:
            True if an endpoint is detected AND we were previously speaking.
            False otherwise.
        """
        if not self._is_speaking:
            # We weren't speaking, so there's nothing to "end"
            return False

        frame_duration_ms = (FRAME_SIZE / SAMPLE_RATE) * 1000
        silence_ms = self._silence_frames * frame_duration_ms

        if silence_ms >= min_silence_ms:
            # Endpoint detected — speaker has paused long enough
            self._is_speaking = False
            return True

        return False

    def reset(self):
        """Reset the VAD state.

        Call this when starting a new recording session or when you
        want to clear any accumulated state from previous audio.
        This resets both the internal Silero model state and our
        speech/silence tracking.
        """
        self.model.reset_states()
        self._is_speaking = False
        self._silence_frames = 0

    # ─── Utility Methods ──────────────────────────────────────────────

    @staticmethod
    def split_into_frames(audio: np.ndarray) -> list[np.ndarray]:
        """Split a longer audio chunk into VAD-sized frames.

        The live pipeline receives audio in chunks (e.g., 0.5 seconds
        = 8000 samples).  But the VAD processes frames of exactly
        FRAME_SIZE (512) samples.  This method splits the chunk
        into frame-sized pieces.

        Any leftover samples that don't fill a complete frame are
        DISCARDED.  This is fine because the next chunk will start
        from a fresh frame boundary.

        Args:
            audio: A 1-D numpy array of float32 audio samples.

        Returns:
            A list of numpy arrays, each of length FRAME_SIZE.

        Example:
            8000 samples → 15 frames of 512 + 320 discarded
        """
        num_frames = len(audio) // FRAME_SIZE
        frames = []

        for i in range(num_frames):
            start = i * FRAME_SIZE
            end = start + FRAME_SIZE
            frames.append(audio[start:end])

        return frames

    def filter_speech(
        self,
        audio_chunk: np.ndarray,
        min_silence_ms: float = 500.0,
    ) -> tuple[np.ndarray | None, bool]:
        """Filter an audio chunk, keeping only speech frames.

        This is a CONVENIENCE method that combines frame splitting,
        speech detection, and endpoint checking into one call.  The
        live pipeline can use this instead of manually iterating frames.

        Args:
            audio_chunk: Raw audio chunk from the microphone.
            min_silence_ms: Silence threshold for endpoint detection.

        Returns:
            A tuple of:
            - speech_audio: A numpy array containing ONLY the speech
              frames concatenated together, or None if no speech was
              detected in this chunk.
            - endpoint: True if an endpoint was detected during this
              chunk (speaker stopped talking).
        """
        frames = self.split_into_frames(audio_chunk)

        if not frames:
            return None, False

        speech_frames = []
        endpoint = False

        for frame in frames:
            result = self.process_frame(frame)

            if result["is_speech"]:
                speech_frames.append(frame)

            # Check for endpoint after each frame
            if self.check_endpoint(min_silence_ms):
                endpoint = True

        if speech_frames:
            return np.concatenate(speech_frames), endpoint
        else:
            return None, endpoint
