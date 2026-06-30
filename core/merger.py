"""
Optimised MedScribe — Transcript Merger (Speaker ↔ Text Alignment)

This module answers: "WHO said WHAT and WHEN?"

At this point in our pipeline, we have two separate outputs:

    From Whisper (transcriber.py):
        [
            {"start": 0.5,  "end": 3.2,  "text": "Aapka naam kya hai?"},
            {"start": 3.5,  "end": 7.8,  "text": "Mera naam Rahul hai."},
            ...
        ]

    From Pyannote (diarizer.py):
        [
            {"start": 0.3,  "end": 3.4,  "speaker": "Doctor"},
            {"start": 3.3,  "end": 8.1,  "speaker": "Patient"},
            ...
        ]

Notice:
    - The timestamps DON'T match exactly.  Whisper says the first sentence
      runs from 0.5s to 3.2s, but Pyannote says the Doctor spoke from 0.3s
      to 3.4s.  These are DIFFERENT models with different internal clocks.
    - There can be OVERLAP between Pyannote segments (two people talking
      at the same time).

This merger module solves the alignment problem using an OVERLAP algorithm:
    For each Whisper text segment, we find the Pyannote speaker segment
    that overlaps with it the MOST (measured in seconds).  That speaker
    "owns" the text.

Then it formats the aligned data into either:
    - Plain text (.txt) — human-readable transcript
    - JSON (.json)      — structured data for SOAP note conversion
"""

import json
from datetime import datetime
from pathlib import Path

from utils.formatting import format_timestamp


# ═══════════════════════════════════════════════════════════════════════
# ALIGNMENT LOGIC
# ═══════════════════════════════════════════════════════════════════════

def find_speaker(
    text_segment: dict,
    speaker_segments: list[dict],
    search_start: int = 0,
) -> tuple[str, int]:
    """Find which speaker is talking during a given text segment.

    This is the core alignment algorithm.  For a text segment with
    timestamps [start, end], we scan through the speaker segments and
    find the one with the MAXIMUM time overlap.

    Visual Example:
        Text segment:     |-------- "Hello doctor" --------|
        Time:             0.5s                             3.2s

        Speaker segments:
        Doctor:    |==========================|
                   0.3s                       3.4s          overlap = 2.7s ← WINNER

        Patient:                        |==========================|
                                        3.3s                       8.1s    overlap = 0.0s

    Optimization — search_start:
        Since both lists are sorted by time, we don't need to search
        from the beginning every time.  We skip past speaker segments
        that have already ended before our text segment starts.  This
        turns an O(n*m) algorithm into roughly O(n+m).

    Args:
        text_segment:    A dict with "start" and "end" keys (from Whisper).
        speaker_segments: The full sorted list of speaker dicts (from Pyannote).
        search_start:    Index to start searching from (optimization).

    Returns:
        A tuple of:
            - The speaker name (e.g., "Doctor") or "Unknown" if no match.
            - The updated search_start index for the next call.
    """
    seg_start = text_segment["start"]
    seg_end = text_segment["end"]

    # Skip speaker segments that ended before this text segment starts.
    # This is the optimization — we never re-scan old segments.
    while search_start < len(speaker_segments) and speaker_segments[search_start]["end"] < seg_start:
        search_start += 1

    best_speaker = "Unknown"
    best_overlap = 0.0
    cursor = search_start

    # Scan forward through speaker segments that could possibly overlap
    while cursor < len(speaker_segments) and speaker_segments[cursor]["start"] <= seg_end:
        spk_seg = speaker_segments[cursor]

        # Calculate overlap duration:
        #   overlap_start = max(text_start, speaker_start)
        #   overlap_end   = min(text_end, speaker_end)
        #   overlap       = overlap_end - overlap_start (if positive)
        overlap_start = max(seg_start, spk_seg["start"])
        overlap_end = min(seg_end, spk_seg["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = spk_seg["speaker"]

        cursor += 1

    # Fallback: if no overlap was found (edge case), use the midpoint
    # method — find the speaker segment that contains the midpoint of
    # the text segment.
    if best_overlap == 0.0:
        midpoint = (seg_start + seg_end) / 2.0
        for spk_seg in speaker_segments[search_start:]:
            if spk_seg["start"] <= midpoint <= spk_seg["end"]:
                best_speaker = spk_seg["speaker"]
                break

    return best_speaker, search_start


# ═══════════════════════════════════════════════════════════════════════
# TRANSCRIPT BUILDING
# ═══════════════════════════════════════════════════════════════════════

def merge_transcript(
    text_segments: list[dict],
    speaker_segments: list[dict] | None = None,
) -> list[dict]:
    """Merge Whisper text segments with Pyannote speaker segments.

    This combines the two data sources into a single unified transcript.
    Each entry in the result has: start, end, speaker, text.

    If speaker_segments is None or empty (diarization was disabled),
    all text segments are assigned the speaker "Speaker".

    Args:
        text_segments:    List of dicts from Whisper: [{"start", "end", "text"}]
        speaker_segments: List of dicts from Pyannote: [{"start", "end", "speaker"}]
                          Can be None if diarization is disabled.

    Returns:
        A unified list of dicts:
        [{"start": 0.5, "end": 3.2, "speaker": "Doctor", "text": "Hello"}]
    """
    merged = []
    search_start = 0

    for text_seg in text_segments:
        if speaker_segments:
            speaker, search_start = find_speaker(
                text_seg, speaker_segments, search_start
            )
        else:
            speaker = "Speaker"

        merged.append({
            "start": text_seg["start"],
            "end": text_seg["end"],
            "speaker": speaker,
            "text": text_seg["text"],
        })

    return merged


def group_by_speaker(merged_segments: list[dict]) -> list[dict]:
    """Group consecutive segments from the same speaker into blocks.

    Without grouping:
        [00:01] Doctor: Aapka naam kya hai?
        [00:04] Doctor: Aur koi problem?
        [00:07] Patient: Mera naam Rahul hai.

    With grouping:
        [00:01] Doctor: Aapka naam kya hai? Aur koi problem?
        [00:07] Patient: Mera naam Rahul hai.

    Grouping makes the transcript much more natural and readable,
    similar to how a human would format a conversation.

    Args:
        merged_segments: The output of merge_transcript().

    Returns:
        A list of grouped segment dicts where consecutive same-speaker
        entries are merged.  Each dict has:
            - "start": Start time of the first segment in the group.
            - "end": End time of the last segment in the group.
            - "speaker": The speaker name.
            - "text": All text joined with a single space.
    """
    if not merged_segments:
        return []

    grouped = []
    current_group = {
        "start": merged_segments[0]["start"],
        "end": merged_segments[0]["end"],
        "speaker": merged_segments[0]["speaker"],
        "text": merged_segments[0]["text"],
    }

    for segment in merged_segments[1:]:
        if segment["speaker"] == current_group["speaker"]:
            # Same speaker — extend the current group
            current_group["end"] = segment["end"]
            current_group["text"] += " " + segment["text"]
        else:
            # Different speaker — save current group, start new one
            grouped.append(current_group)
            current_group = {
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"],
                "text": segment["text"],
            }

    # Don't forget the last group
    grouped.append(current_group)
    return grouped


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTERS
# ═══════════════════════════════════════════════════════════════════════

def format_as_text(
    grouped_segments: list[dict],
    source_file: str,
    language: str,
    model_name: str,
    diarization_enabled: bool,
) -> str:
    """Format the merged transcript as a human-readable plain text string.

    Output looks like:
        ==================================================
          OPTIMISED MEDSCRIBE AI TRANSCRIPT
        ==================================================
          Source File  : D:\\audio\\checkup.wav
          Processed On : 2026-06-15 17:00:00
          ...
        ==================================================

        [00:01] Doctor:
        Aapka naam kya hai? Aur koi problem hai?

        [00:07] Patient:
        Mera naam Rahul hai. Sar mein dard hai.

    Args:
        grouped_segments: Output of group_by_speaker().
        source_file: Path to the original audio file.
        language: Language code used (e.g., "hi").
        model_name: Whisper model variant used.
        diarization_enabled: Whether diarization was used.

    Returns:
        A single string ready to be written to a .txt file.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diarization_status = "Enabled" if diarization_enabled else "Disabled"
    separator = "=" * 50

    # Build the header
    lines = [
        separator,
        "  OPTIMISED MEDSCRIBE AI TRANSCRIPT",
        separator,
        f"  Source File  : {source_file}",
        f"  Processed On : {timestamp_str}",
        f"  Language     : {language}",
        f"  Model        : {model_name}",
        f"  Diarization  : {diarization_status}",
        separator,
        "",  # blank line after header
    ]

    # Build the body
    for segment in grouped_segments:
        time_tag = format_timestamp(segment["start"])

        if diarization_enabled:
            # With speaker labels: show speaker name on a separate line
            lines.append(f"{time_tag} {segment['speaker']}:")
            lines.append(f"{segment['text']}")
            lines.append("")  # blank line between speaker blocks
        else:
            # Without diarization: just timestamp + text
            lines.append(f"{time_tag} {segment['text']}")

    return "\n".join(lines)


def format_as_json(
    grouped_segments: list[dict],
    source_file: str,
    language: str,
    model_name: str,
    diarization_enabled: bool,
) -> dict:
    """Format the merged transcript as a structured JSON dictionary.

    This format is designed to be consumed by downstream systems,
    especially the SOAP note generator that will be built later.

    Output structure:
        {
            "metadata": {
                "source_file": "...",
                "processed_on": "...",
                "language": "hi",
                "model": "distil-large-v3",
                "diarization": true,
                "num_speakers": 2,
                "num_segments": 15
            },
            "speakers": ["Doctor", "Patient"],
            "transcript": [
                {
                    "start": 0.5,
                    "end": 3.2,
                    "start_formatted": "[00:00]",
                    "end_formatted": "[00:03]",
                    "speaker": "Doctor",
                    "text": "Aapka naam kya hai?"
                },
                ...
            ]
        }

    Args:
        grouped_segments: Output of group_by_speaker().
        source_file: Path to the original audio file.
        language: Language code used.
        model_name: Whisper model variant used.
        diarization_enabled: Whether diarization was used.

    Returns:
        A dictionary ready to be serialized with json.dumps().
    """
    # Extract unique speakers in the order they appear
    speakers_seen = []
    for seg in grouped_segments:
        if seg["speaker"] not in speakers_seen:
            speakers_seen.append(seg["speaker"])

    # Build structured transcript entries
    transcript_entries = []
    for seg in grouped_segments:
        transcript_entries.append({
            "start": seg["start"],
            "end": seg["end"],
            "start_formatted": format_timestamp(seg["start"]),
            "end_formatted": format_timestamp(seg["end"]),
            "speaker": seg["speaker"],
            "text": seg["text"],
        })

    return {
        "metadata": {
            "source_file": source_file,
            "processed_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language": language,
            "model": model_name,
            "diarization": diarization_enabled,
            "num_speakers": len(speakers_seen),
            "num_segments": len(transcript_entries),
        },
        "speakers": speakers_seen,
        "transcript": transcript_entries,
    }


# ═══════════════════════════════════════════════════════════════════════
# FILE SAVING
# ═══════════════════════════════════════════════════════════════════════

def save_transcript(
    content: str | dict,
    output_path: str,
    output_format: str = "txt",
) -> str:
    """Save the formatted transcript to a file.

    Args:
        content: Either a string (for txt) or a dict (for json).
        output_path: Full path to the output file (without extension).
        output_format: Either "txt" or "json".

    Returns:
        The full path to the saved file.
    """
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        full_path = f"{output_path}.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    else:
        full_path = f"{output_path}.txt"
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"\n--- Transcript saved → {full_path} ---")
    return full_path
