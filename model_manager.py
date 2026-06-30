"""
Optimised MedScribe — Model Manager (Singleton Model Loader)

This is the CENTRAL HUB for all heavy ML models in the application.

THE PROBLEM IT SOLVES:
    Loading a Whisper model takes 5-15 seconds and uses 1-4 GB of RAM/VRAM.
    Loading Pyannote takes another 5-10 seconds.
    If every file we process creates new model instances, we waste time and
    memory reloading the same weights over and over.

    In the old MedScribe v1.0:
        - Transcriber loaded Whisper in __init__()
        - Diarizer loaded Pyannote in __init__()
        - Processing 5 files = 5 Whisper loads + 5 Pyannote loads = SLOW

THE SOLUTION (Singleton Pattern):
    The ModelManager loads each model ONCE and keeps it alive in memory.
    Every component (transcriber, diarizer, pipeline) asks the ModelManager
    for a model reference instead of loading its own.

    Processing 5 files = 1 Whisper load + 1 Pyannote load = FAST

    This is especially critical for the FastAPI server mode, where the
    server starts up, loads models once, and then serves hundreds of
    requests without ever reloading.

WHAT IS A SINGLETON?
    A Singleton is a design pattern where only ONE instance of a class
    can exist in the entire application.  No matter how many times you
    call ModelManager(), you always get back the SAME object with the
    SAME pre-loaded models.

    We implement this using a class variable (_instance) that stores
    the single instance.  The __new__() method checks if an instance
    already exists before creating a new one.

LAZY LOADING:
    Models are NOT loaded when the ModelManager is created.  They are
    loaded the FIRST time someone asks for them (via get_whisper_model()
    or get_diarization_pipeline()).  This means:
        - If you only use transcription (no diarization), Pyannote is
          never loaded, saving ~2 GB of RAM.
        - The application starts up instantly and loads models on demand.
"""

import os
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config import AppConfig

# Prevent the OpenMP runtime error on Windows when both PyTorch and
# NumPy try to load their own copy of the OpenMP library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class ModelManager:
    """Singleton that loads and caches all heavy ML models.

    Usage:
        config = AppConfig()
        manager = ModelManager(config)

        # First call loads the model (5-15 seconds)
        whisper = manager.get_whisper_model()

        # Second call returns the SAME object instantly
        whisper_again = manager.get_whisper_model()

        assert whisper is whisper_again  # True — same object in memory

    Models managed:
        1. Whisper (faster-whisper) — Speech-to-text
        2. Pyannote Pipeline        — Speaker diarization

    Attributes:
        config: The AppConfig instance with all settings.
        _whisper_model: Cached WhisperModel instance (or None if not loaded).
        _diarization_pipeline: Cached Pyannote Pipeline (or None if not loaded).
    """

    # ─── Singleton Implementation ─────────────────────────────────────
    #
    # _instance stores the single ModelManager that exists in the app.
    # __new__() is called BEFORE __init__() when you do ModelManager().
    # If _instance already exists, __new__() returns it instead of
    # creating a new object.

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Ensure only one ModelManager instance exists (Singleton).

        How this works step by step:
            1. First call:  cls._instance is None → create a new object,
               store it in cls._instance, return it.
            2. Second call: cls._instance is NOT None → skip creation,
               return the existing object.

        This means __init__() will run every time ModelManager() is
        called, but it will always run on the SAME object.  We handle
        this in __init__() by checking if models are already loaded.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: AppConfig | None = None):
        """Initialize the ModelManager.

        Because of the Singleton pattern, this may be called multiple
        times on the same object.  We use a flag (_initialized) to
        ensure we only set up the manager once.

        Args:
            config: The AppConfig instance.  Required on the first call.
                    Ignored on subsequent calls (the original config is kept).

        Raises:
            ValueError: If no config is provided on the first call.
        """
        # Guard: only initialize once, even if __init__ runs again
        if hasattr(self, "_initialized") and self._initialized:
            return

        if config is None:
            raise ValueError(
                "ModelManager requires an AppConfig on first initialization. "
                "Example: ModelManager(AppConfig())"
            )

        self.config = config

        # These start as None — models are loaded lazily on first use
        self._whisper_model: WhisperModel | None = None
        self._diarization_pipeline: Pipeline | None = None

        self._initialized = True
        print("--- ModelManager initialized (models will load on demand) ---")

    # ─── Public API: Model Getters ────────────────────────────────────

    def get_whisper_model(self, model_size: str | None = None) -> WhisperModel:
        """Get the Whisper model, loading it if necessary.

        On first call:
            - Downloads the model weights (if not cached by faster-whisper).
            - Loads the model into RAM/VRAM.
            - This takes 5-15 seconds depending on model size and device.

        On subsequent calls:
            - Returns the already-loaded model instantly.

        Args:
            model_size: Override the model size from config.  Useful for
                        loading a different model for live mode (e.g., "small"
                        for live vs "distil-large-v3" for batch).
                        If None, uses config.model_size.

        Returns:
            A loaded faster_whisper.WhisperModel instance.
        """
        size = model_size or self.config.model_size

        # If we already have a loaded model of the right size, return it
        if self._whisper_model is not None:
            # Check if the requested size matches what's loaded
            # If different size requested, we need to reload
            if hasattr(self, "_loaded_whisper_size") and self._loaded_whisper_size == size:
                return self._whisper_model

            # Different size requested — unload old model first
            print(f"--- Switching Whisper model: {self._loaded_whisper_size} → {size} ---")
            self._unload_whisper()

        print(f"--- Loading Whisper model: {size} ---")
        print(f"    Device: {self.config.device} | Compute: {self.config.compute_type}")

        self._whisper_model = WhisperModel(
            size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

        # Remember what size we loaded so we can detect size changes
        self._loaded_whisper_size = size

        print(f"--- Whisper model loaded successfully ---")
        return self._whisper_model

    def get_diarization_pipeline(self) -> Pipeline:
        """Get the Pyannote diarization pipeline, loading it if necessary.

        On first call:
            - Downloads the pipeline from HuggingFace (requires HF_TOKEN).
            - Loads the neural network weights.
            - Moves the pipeline to the configured device (CPU/GPU).
            - This takes 5-10 seconds.

        On subsequent calls:
            - Returns the already-loaded pipeline instantly.

        Returns:
            A loaded pyannote.audio.Pipeline instance.

        Raises:
            ValueError: If HF_TOKEN is not set in the config or .env file.
        """
        if self._diarization_pipeline is not None:
            return self._diarization_pipeline

        # ── Validate HF_TOKEN ────────────────────────────────────────
        if not self.config.hf_token:
            raise ValueError(
                "HF_TOKEN is required for speaker diarization.\n"
                "Steps to fix:\n"
                "  1. Go to https://huggingface.co/settings/tokens\n"
                "  2. Create a token with 'read' permission\n"
                "  3. Accept the model license at:\n"
                "     https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  4. Add HF_TOKEN=your_token_here to your .env file"
            )

        print("--- Loading Pyannote Diarization Pipeline ---")
        print(f"    Device: {self.config.device}")

        # Download and load the pipeline from HuggingFace
        self._diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.config.hf_token,
        )

        # Move the pipeline to the configured device (CPU or CUDA)
        device = torch.device(self.config.device)
        self._diarization_pipeline.to(device)

        print("--- Pyannote pipeline loaded successfully ---")
        return self._diarization_pipeline

    # ─── Model Lifecycle Management ──────────────────────────────────

    def _unload_whisper(self):
        """Unload the Whisper model from memory.

        This is called internally when switching model sizes.
        Deleting the reference and running garbage collection allows
        Python and CUDA to free the VRAM/RAM.
        """
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            self._loaded_whisper_size = None

            # If using GPU, clear the CUDA memory cache
            if self.config.device == "cuda":
                torch.cuda.empty_cache()

            print("--- Whisper model unloaded ---")

    def _unload_diarization(self):
        """Unload the Pyannote pipeline from memory."""
        if self._diarization_pipeline is not None:
            del self._diarization_pipeline
            self._diarization_pipeline = None

            if self.config.device == "cuda":
                torch.cuda.empty_cache()

            print("--- Pyannote pipeline unloaded ---")

    def unload_all(self):
        """Unload all models and free memory.

        Call this when the application is shutting down or when you
        need to reclaim all GPU/RAM memory.
        """
        self._unload_whisper()
        self._unload_diarization()
        print("--- All models unloaded ---")

    def status(self) -> dict:
        """Get the current status of all managed models.

        Useful for debugging and for the /status endpoint in server mode.

        Returns:
            A dictionary showing which models are loaded and their details.
        """
        return {
            "whisper": {
                "loaded": self._whisper_model is not None,
                "model_size": getattr(self, "_loaded_whisper_size", None),
                "device": self.config.device,
                "compute_type": self.config.compute_type,
            },
            "diarization": {
                "loaded": self._diarization_pipeline is not None,
                "device": self.config.device,
                "hf_token_set": bool(self.config.hf_token),
            },
        }

    # ─── Convenience Factory Methods ─────────────────────────────────
    #
    # These methods create fully-configured component instances.
    # Instead of the pipeline doing:
    #     model = manager.get_whisper_model()
    #     transcriber = OptimisedTranscriber(model, config)
    #
    # It can do:
    #     transcriber = manager.create_transcriber()
    #
    # This keeps the wiring logic in one place.

    def create_transcriber(self):
        """Create a fully configured OptimisedTranscriber instance.

        Returns:
            An OptimisedTranscriber with a pre-loaded Whisper model.
        """
        # Lazy import to avoid circular dependencies and heavy imports
        # at module level.  core.transcriber imports config, and config
        # does not import model_manager, so there's no actual circle —
        # but lazy importing is still good practice for clarity.
        from core.transcriber import OptimisedTranscriber

        model = self.get_whisper_model()
        return OptimisedTranscriber(model=model, config=self.config)

    def create_diarizer(self, speaker_map: dict[str, str] | None = None):
        """Create a fully configured OptimisedDiarizer instance.

        Args:
            speaker_map: Optional custom speaker name mapping.

        Returns:
            An OptimisedDiarizer with a pre-loaded Pyannote pipeline.
        """
        from core.diarizer import OptimisedDiarizer

        pipeline = self.get_diarization_pipeline()
        return OptimisedDiarizer(pipeline=pipeline, speaker_map=speaker_map)

    # ─── Reset (for testing) ─────────────────────────────────────────

    @classmethod
    def reset(cls):
        """Destroy the singleton instance.

        This is primarily used in unit tests where you need a fresh
        ModelManager for each test case.  In production code, you
        should never need to call this.
        """
        if cls._instance is not None:
            cls._instance.unload_all()
            cls._instance = None
            print("--- ModelManager singleton reset ---")
