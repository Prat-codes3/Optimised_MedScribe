"""
Core AI Models Package.

This package contains the wrappers for the heavy machine learning models
used in the project. Each wrapper abstracts away the complex inference logic
and exposes a clean API returning standard Python dictionaries.

Modules:
    transcriber : Whisper model wrapper (Batched Inference).
    diarizer    : Pyannote model wrapper (Speaker Identification).
    vad         : Silero VAD wrapper (Voice Activity Detection).
    agreement   : Local agreement buffer for live transcription.
    merger      : Logic to align speakers to text and format outputs.
"""
