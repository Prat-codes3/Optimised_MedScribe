"""
Optimised MedScribe — Centralized Configuration

Every setting for the entire application lives in this one file.
All other modules import AppConfig instead of defining their own defaults.

Priority order when resolving a value:
    1. CLI argument (highest priority — user typed it explicitly)
    2. .env file variable (user set it once as a preference)
    3. Default value defined below (sensible fallback)
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env file into os.environ so that os.getenv() can read it.
# This runs once when config.py is first imported.
load_dotenv()


def _resolve_device(requested: str) -> str:
    """Determine the best available compute device.

    Args:
        requested: One of "auto", "cpu", or "cuda".

    Returns:
        "cuda" if a compatible NVIDIA GPU is available and requested,
        otherwise "cpu".
    """
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ModuleNotFoundError:
        # If PyTorch itself isn't installed, we can only use CPU.
        print("Warning: PyTorch not found. Falling back to CPU.")
        return "cpu"

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    # requested == "cuda"
    if not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return "cuda"


def _resolve_compute_type(device: str, requested: str) -> str:
    """Picks the best numeric precision for the resolved device.

    Args:
        device: The resolved device string ("cpu" or "cuda").
        requested: One of "auto", "float16", "int8_float16", or "int8".

    Returns:
        A compute-type string that faster-whisper understands.

    Why this matters:
        - float16:      Full half-precision. Fast on GPU, but uses more VRAM.
        - int8_float16: 8-bit weights with 16-bit activations. Best balance
                        of speed and VRAM savings on GPU.
        - int8:         Full 8-bit. Only option on CPU (float16 is not
                        supported on most CPUs).
    """
    if requested != "auto":
        return requested

    # Auto-detect the best compute type for the device.
    if device == "cuda":
        return "int8_float16"  # 50% less VRAM than float16, ~30% faster
    return "int8"  # Only viable option for CPU



# Hinglish prompt 
# This text is fed to Whisper as an "initial_prompt" to bias its predictions
# toward Hindi-English medical vocabulary.

HINGLISH_PROMPT = (
    "This is a Hinglish medical conversation involving Hindi and English. "
    "Use words like 'achha', 'theek hai', 'doctor', 'patient', 'dawayi', "
    "'treatment', 'checkup', 'bukhaar', and 'dawai' when appropriate."
)


@dataclass
class AppConfig:
    """Single source of truth for every tunable setting in the application.

    Attributes are grouped into logical sections: Model, Device, Batch,
    Live, Server, and Paths.  Each attribute has a sensible default that
    can be overridden via environment variables or CLI arguments.
    """

    model_size: str = field(default_factory=lambda: os.getenv("MODEL", "distil-large-v3"))
    language: str = field(default_factory=lambda: os.getenv("LANGUAGE", "hi"))
    beam_size: int = 5
    initial_prompt: str = HINGLISH_PROMPT

    # ── Device ─────────────────────────────────────────────────────────
    # Raw device preference before resolution.  Call resolve() to get
    # the actual usable device and compute type.
    device_preference: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    compute_type_preference: str = field(default_factory=lambda: os.getenv("COMPUTE_TYPE", "auto"))

    # ── Batch Pipeline ─────────────────────────────────────────────────
    batch_size: int = 8  # Number of 30s windows processed per GPU call
    diarize: bool = False
    output_format: str = "txt"  # "txt" or "json"

    # ── Live Pipeline ──────────────────────────────────────────────────
    live_model: str = field(default_factory=lambda: os.getenv("LIVE_MODEL", "small"))

    live_interval: float = 2.0  # Seconds of new audio before each pass
    live_overlap: float = 0.75  # Seconds of trailing context between passes
    live_beam_size: int = 1  # Greedy decoding for lowest latency
    use_vad: bool = True  # Silero VAD gate to skip silence
    use_agreement: bool = True  # Local Agreement to prevent duplicates
    vad_threshold: float = 0.5  # Confidence threshold for speech detection

    # ── Server ─────────────────────────────────────────────────────────
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    # ── Paths / Auth ───────────────────────────────────────────────────
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    # ── Resolved (set after calling resolve()) ─────────────────────────
    device: str = field(default="cpu", init=False)
    compute_type: str = field(default="int8", init=False)

    def __post_init__(self):
        """Automatically resolve device and compute type after init."""
        self.resolve()

    def resolve(self):
        """Resolve device and compute type based on preferences and hardware.

        This method can be called again if preferences change at runtime.
        """
        self.device = _resolve_device(self.device_preference)
        self.compute_type = _resolve_compute_type(self.device, self.compute_type_preference)

    def summary(self) -> str:
        """Return a human-readable summary of the active configuration."""
        lines = [
            "╔══════════════════════════════════════╗",
            "║     Optimised MedScribe Config       ║",
            "╠══════════════════════════════════════╣",
            f"║  Device:       {self.device.upper():<21}║",
            f"║  Compute Type: {self.compute_type:<21}║",
            f"║  Model:        {self.model_size:<21}║",
            f"║  Language:     {self.language:<21}║",
            f"║  Beam Size:    {self.beam_size:<21}║",
            f"║  Batch Size:   {self.batch_size:<21}║",
            f"║  Diarization:  {'Enabled' if self.diarize else 'Disabled':<21}║",
            f"║  Output:       {self.output_format.upper():<21}║",
            f"║  HF Token:     {'Set' if self.hf_token else 'Not Set':<21}║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)
