import json
from dataclasses import dataclass
from enum import Enum

import librosa
import numpy as np
from numpy.typing import NDArray


class ModelType(Enum):
    KICKDRUM = "Kickdrum"
    SNAREDRUM = "Snaredrum"


@dataclass
class ModelParams:
    # Model parameters
    DEVICE = "cpu"
    LATENT_DIM = 128
    BATCH_SIZE = 32

    # Training parameters
    CRITIC_STEPS = 5
    LR_G = 2e-4
    LR_C = 4e-4
    LAMBDA_GP = 10
    N_EPOCHS = 50
    PATIENCE = 10

    # PG-GAN parameters
    MAX_STAGE = 4
    INITIAL_SIZE = 16
    GROWTH_FACTOR = 2.0

    # Model specific parameters
    def __init__(self):
        self.outputs_dir = "outputs"
        self.compiled_data_path = "data/compiled_data.npy"
        self.generated_audio_name = "generated_audio"
        self.visualize_generated = True
        self.sample_length: float = 0.0
        self.model_save_path: str = ""
        self.training_audio_dir: str = ""

    def load_params(self, model_type: ModelType) -> None:
        """Load JSON params for the specified model type."""
        desired_model = model_type.value
        with open("src/utils/model_params.json", "r") as f:
            model_data = json.load(f)

        selected_model = None
        for model in model_data:
            if model["model_name"] == desired_model:
                selected_model = model
                break

        if selected_model is None:
            raise ValueError(f"Model '{desired_model}' not found in model_params.json")

        # Set params from JSON
        self.selected_model = selected_model["model_name"]
        self.sample_length = selected_model["sample_length"]
        self.model_save_path = selected_model["model_save_path"]
        self.training_audio_dir = selected_model["train_data_dir"]


class SignalConstants:
    # Signal data dimensions
    SR = 44100
    CHANNELS = 2
    FRAMES = 256
    LINEAR_SPEC_FBINS = 1024
    MEL_SPEC_FBINS = 256
    MEL_MIN_FREQ = 20

    def __init__(self, sample_length: float) -> None:
        self.sample_length = sample_length

    @property
    def MEL_MAX_FREQ(self) -> int:
        return int(self.SR / 2)

    @property
    def WINDOW(self) -> NDArray[np.float32]:
        return librosa.filters.get_window("hann", self.FT_WIN, fftbins=True)

    @property
    def FT_WIN(self) -> int:
        return (self.LINEAR_SPEC_FBINS - 1) * 2

    @property
    def FT_HOP(self) -> int:
        total_samples = int(self.sample_length * self.SR)
        hop = (total_samples - self.FT_WIN) // (self.FRAMES - 1)
        return hop
