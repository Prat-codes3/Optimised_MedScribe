"""
Optimised MedScribe — Speaker Diarizer (Pyannote)

This module answers the question: "WHO is speaking WH  EN?"

Whisper tells us WHAT was said and at what timestamps, but it has no idea
who said it.  Pyannote Audio is a separate neural network that listens to
the audio and identifies distinct speakers based on voice characteristics
(pitch,  speaking style).

How Pyannote works (simplified):
    1. It scans the full audio and detects all voice segments.
    2. It generates a "voice embedding" (a numeric fingerprint) for each
       segment of speech it finds.
    3. It clusters embeddings that sound similar together and assigns them
       a label (SPEAKER_00, SPEAKER_01, etc.).
    4. It returns a timeline: "SPEAKER_00 spoke from 0.5s to 3.2s,
       SPEAKER_01 spoke from 3.5s to 7.8s, ..."

This module wraps all of that behind a clean API and adds:
    - Dependency Injection: Receives a pre-loaded pipeline (no self-loading).
    - JSON Caching: Saves results to disk so re-running skips diarization.
    - Speaker Relabeling: Maps "SPEAKER_00" → "Doctor", "SPEAKER_01" → "Patient".

Design pattern used: Same as transcriber.py — the class does NOT create or
load the heavy ML model itself.  It receives a pre-loaded pipeline from the
ModelManager.  This means we can keep the model warm in memory for the
FastAPI server or for batch processing multiple files.
"""

import json
import hashlib
from pathlib import Path

from pyannote.audio import Pipeline


# ─── Default Speaker Labels ──────────────────────────────────────────────
# In a medical context, the first speaker detected is usually the Doctor
# (who starts the conversation), and the second is the Patient.
# If there are more than 2 speakers (e.g., a nurse, a family member),
# they get generic labels like "Speaker 3", "Speaker 4", etc.
#
# These can be overridden by the caller if needed.

DEFAULT_SPEAKER_MAP = {
    "SPEAKER_00": "Doctor",
    "SPEAKER_01": "Patient",
}


class OptimisedDiarizer:
    """Wrapper for the Pyannote speaker diarization pipeline.

    This class handles speaker identification from audio files.
    It does NOT load its own model — it receives a pre-loaded Pyannote
    Pipeline instance from the ModelManager via dependency injection.

    Attributes:
        pipeline: A pre-loaded pyannote.audio.Pipeline instance.
        speaker_map: A dictionary mapping Pyannote's internal labels
                     (like "SPEAKER_00") to human-readable names.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        speaker_map: dict[str, str] | None = None,
    ):
        """Initialize the diarizer.

        Args:
            pipeline: A pre-loaded pyannote.audio.Pipeline instance.
                      This is created and managed by the ModelManager.
            speaker_map: Optional dictionary to rename speakers.
                         Defaults to {"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}.
        """
        self.pipeline = pipeline
        self.speaker_map = speaker_map or DEFAULT_SPEAKER_MAP

    # ─── Public API ───────────────────────────────────────────────────

    def diarize(
        self,
        audio_path: str,
        cache_dir: str | None = None,
    ) -> list[dict]:
        """Run speaker diarization on an audio file.

        This is the main method you call.  It will:
            1. Check if a cached result exists for this audio file.
            2. If cached → load from JSON (instant, no GPU needed).
            3. If not cached → run Pyannote → save to JSON → return.

        Args:
            audio_path: Absolute path to the audio file to diarize.
            cache_dir:  Optional directory to store/read cached results.
                        If None, caching is disabled.

        Returns:
            A list of speaker segment dictionaries, sorted by start time:
            [
                {"start": 0.5, "end": 3.2, "speaker": "Doctor"},
                {"start": 3.5, "end": 7.8, "speaker": "Patient"},
                ...
            ]
        """
        # ── Step 1: Try loading from cache ────────────────────────────
        if cache_dir:
            cached = self._load_cache(audio_path, cache_dir)
            if cached is not None:
                print(f"--- Diarization loaded from cache ({len(cached)} segments) ---")
                return cached

        # ── Step 2: Run Pyannote diarization ──────────────────────────
        print("--- Running Speaker Diarization (Pyannote 3.1) ---")

        # This is the heavy computation — Pyannote processes the full audio.
        # On GPU, a 5-minute file takes ~10-15 seconds.
        # On CPU, the same file can take 2-5 minutes.
        diarization_result = self.pipeline(audio_path)

        # ── Step 3: Extract and clean speaker segments ────────────────
        speaker_segments = self._extract_segments(diarization_result)

        print(f"--- Diarization complete: {len(speaker_segments)} segments, "
              f"{self._count_speakers(speaker_segments)} speakers ---")

        # ── Step 4: Save to cache for next time ───────────────────────
        if cache_dir:
            self._save_cache(audio_path, cache_dir, speaker_segments)

        return speaker_segments

    # ─── Internal Helpers ─────────────────────────────────────────────

    def _extract_segments(self, diarization_result) -> list[dict]:
        """Convert raw Pyannote output into clean dictionaries.

        Pyannote returns an Annotation object.  We iterate over it using
        .itertracks(yield_label=True) which gives us:
            - turn: A Segment object with .start and .end (float seconds)
            - _: A track name (we don't need this)
            - speaker: The raw speaker label like "SPEAKER_00"

        We then:
            1. Map the raw label to a human name using self.speaker_map.
            2. Round timestamps to 3 decimal places (millisecond precision).
            3. Sort by start time so the output is chronological.

        Args:
            diarization_result: The raw Annotation object from Pyannote.

        Returns:
            Sorted list of segment dictionaries.
        """
        segments = []

        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            # Map "SPEAKER_00" → "Doctor", or keep original if not in map
            label = self._get_speaker_label(speaker)

            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": label,
            })

        # Sort by start time, then by end time (for ties)
        segments.sort(key=lambda seg: (seg["start"], seg["end"]))
        return segments

    def _get_speaker_label(self, raw_label: str) -> str:
        """Map a raw Pyannote label to a human-readable name.

        Lookup order:
            1. Check self.speaker_map (e.g., "SPEAKER_00" → "Doctor")
            2. If not found, extract the number and create a generic label
               (e.g., "SPEAKER_04" → "Speaker 5")

        Args:
            raw_label: The raw label from Pyannote (e.g., "SPEAKER_00").

        Returns:
            A human-readable speaker name.
        """
        # Check the user-provided (or default) mapping first
        if raw_label in self.speaker_map:
            return self.speaker_map[raw_label]

        # Extract the number from "SPEAKER_XX" and create a 1-indexed label
        # "SPEAKER_02" → 2 → "Speaker 3"
        try:
            number = int(raw_label.split("_")[-1])
            return f"Speaker {number + 1}"
        except (ValueError, IndexError):
            # If the label format is unexpected, return it as-is
            return raw_label

    def _count_speakers(self, segments: list[dict]) -> int:
        """Count the number of unique speakers in the segments.

        Args:
            segments: List of segment dictionaries.

        Returns:
            Number of unique speakers found.
        """
        return len({seg["speaker"] for seg in segments})

    # ─── Caching System ──────────────────────────────────────────────
    #
    # WHY CACHE?
    # Diarization is the slowest step in our pipeline (2-5 min on CPU).
    # If you process the same audio file twice (e.g., you ran batch mode,
    # then wanted to re-run with different Whisper settings), the speaker
    # segments won't change — only the text might.
    #
    # By caching the diarization result as a JSON file, the second run
    # loads instantly instead of waiting 5 minutes again.
    #
    # HOW DO WE KNOW IT'S THE SAME FILE?
    # We compute an MD5 hash of the audio file's content.  Even if the
    # file is renamed or moved, if the bytes are identical, the hash will
    # match and we'll use the cache.
    #

    def _get_cache_path(self, audio_path: str, cache_dir: str) -> Path:
        """Generate the cache file path for a given audio file.

        The cache filename is based on the MD5 hash of the audio file's
        contents, so even if the file is renamed, the cache still works.

        Args:
            audio_path: Path to the audio file.
            cache_dir: Directory where cache files are stored.

        Returns:
            A Path object pointing to the cache JSON file.
        """
        # Compute MD5 hash of the audio file's content
        file_hash = self._compute_file_hash(audio_path)
        cache_filename = f"diarize_{file_hash}.json"
        return Path(cache_dir) / cache_filename

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of a file's contents.

        We read the file in 8KB chunks to avoid loading a 500MB audio file
        entirely into RAM.

        Args:
            file_path: Path to the file to hash.

        Returns:
            A hexadecimal MD5 hash string (32 characters).
        """
        hasher = hashlib.md5()

        with open(file_path, "rb") as f:
            # Read in 8KB chunks — efficient for large audio files
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)

        return hasher.hexdigest()

    def _load_cache(self, audio_path: str, cache_dir: str) -> list[dict] | None:
        """Try to load cached diarization results.

        Args:
            audio_path: Path to the audio file (used to find its cache).
            cache_dir: Directory where cache files are stored.

        Returns:
            The cached list of speaker segments, or None if no cache exists.
        """
        cache_path = self._get_cache_path(audio_path, cache_dir)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate that the cached data looks correct
            if isinstance(data, list) and all(isinstance(seg, dict) and "start" in seg and "end" in seg and "speaker" in seg for seg in data):
                return data

            # Cache file exists but has invalid structure — ignore it
            print("Warning: Cache file has invalid structure, re-running diarization.")
            return None

        except (json.JSONDecodeError, OSError) as e:
            # Cache file is corrupted or unreadable
            print(f"Warning: Could not read cache file: {e}")
            return None

    def _save_cache(
        self, audio_path: str, cache_dir: str, segments: list[dict]
    ) -> None:
        """Save diarization results to a JSON cache file.

        Args:
            audio_path: Path to the audio file (used to generate cache key).
            cache_dir: Directory where cache files are stored.
            segments: The diarization results to cache.
        """
        cache_path = self._get_cache_path(audio_path, cache_dir)

        # Create cache directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)

            print(f"--- Diarization cached → {cache_path.name} ---")

        except OSError as e:
            # Caching failure is non-fatal — just warn and continue
            print(f"Warning: Could not save cache: {e}")
