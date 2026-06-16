import os
from typing import Iterator

from faster_whisper import BatchedInferencePipeline, WhisperModel
import faster_whisper.transcribe

# Import our config and utility helpers
from config import AppConfig
from utils.formatting import clean_text

# Prevent the OpenMP runtime error on Windows when using PyTorch/NumPy.
# This must run before any heavy ML models do their math.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class OptimisedTranscriber:
    """Wrapper for the faster-whisper model.

    This class handles text extraction from audio using Whisper.
    Unlike v1.0, this class does NOT load its own model. It receives a pre-loaded
    WhisperModel instance from the ModelManager. This allows us to keep the model
    warm in memory for server mode or multi-file batch processing.
    """

    def __init__(self, model: WhisperModel, config: AppConfig):
        """Initialize the transcriber.

        Args:
            model: A pre-loaded faster_whisper.WhisperModel instance.
            config: The centralized AppConfig containing our settings.
        """
        self.model = model
        self.config = config

    def transcribe_batch(self, audio_path: str) -> tuple[list[dict], str]:
        """Transcribe an entire audio file efficiently using GPU batching.

        This is used by the Batch Pipeline. It uses BatchedInferencePipeline,
        which splits the audio into 30-second windows and sends multiple
        windows to the GPU at the exact same time (batch_size). This provides
        a massive speedup (2x-4x) compared to sequential processing.

        Args:
            audio_path: Absolute path to the audio file.

        Returns:
            A tuple containing:
              - A list of segment dictionaries: [{"start": float, "end": float, "text": str}]
              - The detected language string (e.g., "hi").
        """
        print(f"--- Transcribing (Batched, size={self.config.batch_size}) ---")

        # Wrap our model in the batched pipeline
        batched_model = BatchedInferencePipeline(model=self.model)

        # Run inference
        segments_generator, info = batched_model.transcribe(
            audio_path,
            batch_size=self.config.batch_size,
            language=self.config.language,
            beam_size=self.config.beam_size,
            initial_prompt=self.config.initial_prompt,
            vad_filter=True,  # Always use VAD to skip silent regions
        )

        return self._process_segments(segments_generator), info.language

    def transcribe_chunk(self, audio_chunk) -> str:
        """Transcribe a small chunk of audio in memory.

        This is used exclusively by the Live Pipeline. Since we are processing
        live streaming data (e.g., 2 seconds of audio), we cannot use batching.
        We use greedy decoding (beam_size=1) for maximum speed/lowest latency.

        Args:
            audio_chunk: A numpy array containing raw audio data.

        Returns:
            A single string containing the transcribed text for that chunk.
        """
        segments_generator, _ = self.model.transcribe(
            audio_chunk,
            language=self.config.language,
            beam_size=self.config.live_beam_size,  # Greedy search (1) for speed
            initial_prompt=self.config.initial_prompt,
            vad_filter=False,  # VAD is handled externally in live mode
            condition_on_previous_text=False, # Avoid hallucination loops
        )

        # For live chunks, we just want to combine all the text together
        full_text = ""
        for segment in segments_generator:
            text = clean_text(segment.text)
            if text:
                full_text += f"{text} "

        return full_text.strip()

    def _process_segments(self, segments: Iterator) -> list[dict]:
        """Clean and format raw Whisper segments into standard dictionaries.

        Args:
            segments: An iterator of raw Segment objects from faster-whisper.

        Returns:
            A clean list of dictionaries ready for diarization alignment.
        """
        transcript_data = []

        for segment in segments:
            # Use our centralized cleaning utility
            text = clean_text(segment.text)

            # Skip empty segments (sometimes Whisper outputs just punctuation)
            if not text or text in {".", ",", "?", "!"}:
                continue

            transcript_data.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": text,
                }
            )

            # Live terminal feedback
            print(f"[{segment.start:.2f}s] {text}")

        return transcript_data
