"""
Formatting Utilities — Timestamp conversion and text cleanup.

These are simple, pure-Python helper functions.  They have zero external
dependencies (no torch, no numpy).  Every module in the project that needs
to display a timestamp or clean up text will import from here instead of
re-implementing the logic.

Why a separate file?
    In v1.0, format_timestamp() was duplicated in both main.py and
    live_scribe.py.  If we ever wanted to change the format (e.g., add
    hours), we would have to update two places and hope we didn't miss one.
    Centralising it here follows the DRY principle (Don't Repeat Yourself).
"""

from datetime import datetime


def format_timestamp(seconds: float) -> str:
    """Convert a float number of seconds into a human-readable clock string.

    Args:
        seconds: The time position in the audio, in seconds.
                 For example, 125.7 means 2 minutes and 5.7 seconds in.

    Returns:
        A string like "[02:05]" (minutes:seconds).
        If the audio is over an hour, returns "[01:02:05]" (hours:minutes:seconds).

    Examples:
        >>> format_timestamp(0.0)
        '[00:00]'
        >>> format_timestamp(65.5)
        '[01:05]'
        >>> format_timestamp(3725.0)
        '[01:02:05]'
    """
    total_seconds = int(seconds)

    if total_seconds >= 3600:
        # Audio longer than 1 hour — include hours.
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"

    # Standard case — minutes and seconds only.
    minutes, secs = divmod(total_seconds, 60)
    return f"[{minutes:02d}:{secs:02d}]"


def clean_text(text: str) -> str:
    """Strip whitespace and normalize a transcribed text segment.

    Whisper sometimes produces text with leading/trailing spaces, double
    spaces, or invisible Unicode characters.  This function cleans all of
    that up in one place.

    Args:
        text: The raw text string from the transcription model.

    Returns:
        A cleaned string, or an empty string if the input was only whitespace.

    Examples:
        >>> clean_text("  Hello  doctor   saab  ")
        'Hello doctor saab'
        >>> clean_text("   ")
        ''
    """
    # strip()  — removes leading and trailing whitespace.
    # split()  — splits on ANY whitespace (spaces, tabs, newlines) and
    #            removes empty strings caused by double-spaces.
    # " ".join — glues the words back together with exactly one space.
    return " ".join(text.strip().split())


def generate_header(
    source_file: str,
    language: str,
    diarization_enabled: bool,
    model_name: str,
) -> str:
    """Generate the header block for a transcript file.

    This header is placed at the very top of every saved transcript so that
    the reader immediately knows what file was processed, when, and with
    what settings.

    Args:
        source_file: Path to the original audio file.
        language: The language code used (e.g., "hi").
        diarization_enabled: Whether speaker diarization was used.
        model_name: The Whisper model variant used (e.g., "distil-large-v3").

    Returns:
        A multi-line string ready to be written to a file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diarization_status = "Enabled" if diarization_enabled else "Disabled"
    separator = "=" * 50

    return (
        f"{separator}\n"
        f"  OPTIMISED MEDSCRIBE AI TRANSCRIPT\n"
        f"{separator}\n"
        f"  Source File  : {source_file}\n"
        f"  Processed On : {timestamp}\n"
        f"  Language     : {language}\n"
        f"  Model        : {model_name}\n"
        f"  Diarization  : {diarization_status}\n"
        f"{separator}\n\n"
    )
