"""
Optimised MedScribe — Batch Pipeline (File Processing Orchestrator)

This is the CONDUCTOR of the entire batch processing workflow.
It coordinates all the components we've built so far:

    ┌─────────────┐
    │  Audio File  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────┐
    │  validate_audio  │  ← utils/audio.py
    └──────┬──────────┘
           │
           ├──────────────────────────┐
           ▼                          ▼
    ┌──────────────┐          ┌──────────────┐
    │  Transcriber  │          │   Diarizer   │
    │   (Whisper)   │          │  (Pyannote)  │
    └──────┬───────┘          └──────┬───────┘
           │                          │
           │   text_segments          │  speaker_segments
           │                          │
           └──────────┬───────────────┘
                      ▼
               ┌──────────────┐
               │    Merger     │  ← core/merger.py
               └──────┬───────┘
                      │
                      ▼
               ┌──────────────┐
               │  Save Output  │  .txt or .json
               └──────────────┘

WHY A SEPARATE PIPELINE FILE?
    In the old MedScribe v1.0, all of this orchestration logic was crammed
    into main.py's process_audio() function.  That mixed CLI argument parsing,
    model loading, transcription, diarization, merging, and file saving into
    one giant function.

    By separating the pipeline into its own module:
    - main.py becomes a thin CLI parser (its only job is reading user flags).
    - server.py can import and call the same pipeline without duplicating code.
    - Testing becomes easy — you can test the pipeline without the CLI.

SINGLE FILE vs DIRECTORY MODE:
    The pipeline supports two modes:
    1. Single file:  process_file("D:/audio/checkup.wav")
    2. Directory:    process_directory("D:/audio/")  → processes all audio files
"""

import time
import os

from config import AppConfig
from model_manager import ModelManager
from core.merger import (
    merge_transcript,
    group_by_speaker,
    format_as_text,
    format_as_json,
    save_transcript,
)
from utils.audio import validate_audio_file, build_output_path, collect_audio_files


class BatchPipeline:
    """Orchestrates the full batch transcription workflow.

    This class ties together the ModelManager, Transcriber, Diarizer,
    and Merger into a single, clean pipeline.  It is used by both the
    CLI (main.py) and the server (server.py).

    Usage:
        config = AppConfig(diarize=True, output_format="json")
        pipeline = BatchPipeline(config)

        # Process a single file
        result = pipeline.process_file("D:/audio/checkup.wav")

        # Process an entire directory
        results = pipeline.process_directory("D:/audio/")

    Attributes:
        config: The AppConfig with all settings.
        manager: The ModelManager singleton (loads models on demand).
    """

    def __init__(self, config: AppConfig):
        """Initialize the batch pipeline.

        This does NOT load any models yet — they are loaded lazily
        by the ModelManager when first needed.

        Args:
            config: The centralized AppConfig instance.
        """
        self.config = config
        self.manager = ModelManager(config)

    def process_file(self, audio_path: str) -> dict:
        """Process a single audio file end-to-end.

        This is the main method.  It runs through the complete pipeline:
            1. Validate the audio file
            2. Transcribe with Whisper (batched inference)
            3. Diarize with Pyannote (if enabled)
            4. Merge text + speakers
            5. Format and save output

        Args:
            audio_path: Path to the audio file (relative or absolute).

        Returns:
            A result dictionary containing:
            {
                "input_file": str,
                "output_file": str,
                "language": str,
                "num_segments": int,
                "num_speakers": int,
                "duration_seconds": float,
                "success": bool,
                "error": str | None,
            }
        """
        start_time = time.time()

        try:
            # ── Step 1: Validate ──────────────────────────────────────
            print(f"\n{'='*50}")
            print(f"  Processing: {os.path.basename(audio_path)}")
            print(f"{'='*50}")

            absolute_path = validate_audio_file(audio_path)

            # ── Step 2: Transcribe ────────────────────────────────────
            print("\n--- Phase 1: Transcription ---")
            transcriber = self.manager.create_transcriber()
            text_segments, detected_language = transcriber.transcribe_batch(absolute_path)

            if not text_segments:
                print("Warning: No speech detected in the audio file.")
                return self._build_result(
                    audio_path, None, "unknown", 0, 0,
                    time.time() - start_time, True, None,
                )

            print(f"--- Transcription complete: {len(text_segments)} segments ---")

            # ── Step 3: Diarize (optional) ────────────────────────────
            speaker_segments = None

            if self.config.diarize:
                print("\n--- Phase 2: Speaker Identification ---")
                try:
                    diarizer = self.manager.create_diarizer()

                    # Use a cache directory next to the audio file
                    cache_dir = os.path.join(
                        os.path.dirname(absolute_path), ".medscribe_cache"
                    )
                    speaker_segments = diarizer.diarize(
                        absolute_path, cache_dir=cache_dir
                    )
                except ValueError as e:
                    # HF_TOKEN not set — warn but continue without diarization
                    print(f"Warning: Diarization skipped — {e}")
                    speaker_segments = None
                except Exception as e:
                    # Any other diarization error — warn but continue
                    print(f"Warning: Diarization failed ({e}). Continuing without speakers.")
                    speaker_segments = None

            # ── Step 4: Merge ─────────────────────────────────────────
            print("\n--- Phase 3: Building Final Transcript ---")

            merged = merge_transcript(text_segments, speaker_segments)
            grouped = group_by_speaker(merged)

            diarization_enabled = speaker_segments is not None and len(speaker_segments) > 0

            # ── Step 5: Format ────────────────────────────────────────
            if self.config.output_format == "json":
                content = format_as_json(
                    grouped,
                    source_file=absolute_path,
                    language=detected_language,
                    model_name=self.config.model_size,
                    diarization_enabled=diarization_enabled,
                )
            else:
                content = format_as_text(
                    grouped,
                    source_file=absolute_path,
                    language=detected_language,
                    model_name=self.config.model_size,
                    diarization_enabled=diarization_enabled,
                )

            # ── Step 6: Save ──────────────────────────────────────────
            output_base = build_output_path(
                absolute_path, suffix="_transcript", ext=""
            )
            output_file = save_transcript(
                content, output_base, self.config.output_format
            )

            # ── Done ──────────────────────────────────────────────────
            elapsed = time.time() - start_time
            num_speakers = len({seg["speaker"] for seg in grouped})

            print(f"\n--- SUCCESS in {elapsed:.1f}s ---")
            print(f"    Output: {output_file}")

            return self._build_result(
                absolute_path, output_file, detected_language,
                len(grouped), num_speakers, elapsed, True, None,
            )

        except (FileNotFoundError, ValueError) as e:
            # Known errors (bad file path, unsupported format)
            elapsed = time.time() - start_time
            print(f"\n--- ERROR: {e} ---")
            return self._build_result(
                audio_path, None, "unknown", 0, 0, elapsed, False, str(e),
            )

        except Exception as e:
            # Unexpected errors (model crash, out of memory, etc.)
            elapsed = time.time() - start_time
            print(f"\n--- UNEXPECTED ERROR: {e} ---")
            return self._build_result(
                audio_path, None, "unknown", 0, 0, elapsed, False, str(e),
            )

    def process_directory(self, directory_path: str) -> list[dict]:
        """Process all audio files in a directory.

        This scans the directory for supported audio files and processes
        each one sequentially.  Results are collected into a summary.

        Args:
            directory_path: Path to the directory containing audio files.

        Returns:
            A list of result dictionaries (one per file).

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If no supported audio files are found.
        """
        audio_files = collect_audio_files(directory_path)

        print(f"\n{'='*50}")
        print(f"  Batch Processing: {len(audio_files)} files")
        print(f"  Directory: {os.path.abspath(directory_path)}")
        print(f"{'='*50}")

        results = []
        total_start = time.time()

        for index, file_path in enumerate(audio_files, start=1):
            print(f"\n--- [{index}/{len(audio_files)}] ---")
            result = self.process_file(file_path)
            results.append(result)

        total_elapsed = time.time() - total_start

        # Print summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        print(f"\n{'='*50}")
        print(f"  BATCH COMPLETE")
        print(f"{'='*50}")
        print(f"  Total files : {len(results)}")
        print(f"  Successful  : {successful}")
        print(f"  Failed      : {failed}")
        print(f"  Total time  : {total_elapsed:.1f}s")
        print(f"{'='*50}")

        return results

    # ─── Internal Helpers ─────────────────────────────────────────────

    @staticmethod
    def _build_result(
        input_file: str,
        output_file: str | None,
        language: str,
        num_segments: int,
        num_speakers: int,
        duration_seconds: float,
        success: bool,
        error: str | None,
    ) -> dict:
        """Build a standardized result dictionary.

        Having a consistent result format makes it easy for:
        - The CLI to print a summary.
        - The server to return a JSON response.
        - Batch mode to aggregate results.

        Args:
            input_file: Path to the source audio file.
            output_file: Path to the saved transcript (or None on failure).
            language: Detected/used language code.
            num_segments: Number of grouped transcript segments.
            num_speakers: Number of unique speakers found.
            duration_seconds: How long processing took.
            success: Whether the processing completed without error.
            error: Error message string, or None if successful.

        Returns:
            A dictionary with all the result information.
        """
        return {
            "input_file": input_file,
            "output_file": output_file,
            "language": language,
            "num_segments": num_segments,
            "num_speakers": num_speakers,
            "duration_seconds": round(duration_seconds, 2),
            "success": success,
            "error": error,
        }
