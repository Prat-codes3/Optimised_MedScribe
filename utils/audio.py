"""
Audio Utilities — File validation and format helpers.

Before we send an audio file to the AI model, we need to verify:
    1. Does the file actually exist?
    2. Is it a supported audio format?
    3. What is the output path for the transcript?

These checks are done here so that the pipeline modules stay clean and
focused on orchestration, not file system logic.
"""

import os
from pathlib import Path

# Every audio format that faster-whisper (via FFmpeg) can decode.
# If a user passes a .docx or .pdf, we catch it early with a clear error.
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".m4a",
    ".flac",
    ".ogg",
    ".wma",
    ".aac",
    ".webm",
}


def validate_audio_file(file_path: str) -> str:
    """Check that an audio file exists and has a supported format.

    Args:
        file_path: The path to the audio file (relative or absolute).

    Returns:
        The resolved absolute path to the file.

    Raises:
        FileNotFoundError: If the file does not exist on disk.
        ValueError: If the file extension is not in SUPPORTED_EXTENSIONS.

    Why return the absolute path?
        Libraries like FFmpeg and pyannote sometimes spawn subprocesses
        that run from a different working directory.  An absolute path
        guarantees they can always find the file.
    """
    absolute_path = os.path.abspath(file_path)

    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Audio file not found: {absolute_path}")

    extension = Path(absolute_path).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio format '{extension}'. "
            f"Supported formats: {supported_list}"
        )

    return absolute_path


def build_output_path(audio_path: str, suffix: str = "_transcript", ext: str = ".txt") -> str:
    """Generate the output file path based on the input audio file name.

    The output file is placed in the same directory as the input file.

    Args:
        audio_path: Absolute path to the source audio file.
        suffix: Text appended to the base name (default: "_transcript").
        ext: File extension for the output (default: ".txt").

    Returns:
        Absolute path for the output file.

    Examples:
        >>> build_output_path("D:/audio/consult.wav")
        'D:/audio/consult_transcript.txt'
        >>> build_output_path("D:/audio/consult.wav", ext=".json")
        'D:/audio/consult_transcript.json'
    """
    directory = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(directory, f"{base_name}{suffix}{ext}")


def collect_audio_files(directory_path: str) -> list[str]:
    """Find all supported audio files in a directory.

    This is used by the batch pipeline when the user passes --directory
    instead of --file, allowing them to transcribe an entire folder of
    recordings in one command.

    Args:
        directory_path: Path to the directory to scan.

    Returns:
        A sorted list of absolute paths to supported audio files.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no supported audio files are found.
    """
    abs_dir = os.path.abspath(directory_path)

    if not os.path.isdir(abs_dir):
        raise FileNotFoundError(f"Directory not found: {abs_dir}")

    audio_files = []
    for file_name in sorted(os.listdir(abs_dir)):
        extension = Path(file_name).suffix.lower()
        if extension in SUPPORTED_EXTENSIONS:
            audio_files.append(os.path.join(abs_dir, file_name))

    if not audio_files:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"No supported audio files found in {abs_dir}. "
            f"Supported formats: {supported_list}"
        )

    return audio_files
