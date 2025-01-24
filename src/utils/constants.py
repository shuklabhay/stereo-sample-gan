import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import hann


class ModelType(Enum):
    KICKDRUM = "Kickdrum"
    SNARE = "Snare"
    CHORDSHOT = "ChordShot"


model_selection = ModelType.SNARE


@dataclass
class ModelParams:
    LATENT_DIM = 128
    BATCH_SIZE = 16
    DROPOUT_RATE = 0.2

    def __init__(self):
        self.load_params()

        self.outputs_dir = "outputs"
        self.compiled_data_path = "data/compiled_data.npy"
        self.generated_audio_name = "generated_audio"
        self.visualize_generated = True
        self.sample_length: float
        self.model_save_path: str
        self.training_audio_dir: str

    def load_params(self, model_name: ModelType = model_selection) -> None:
        with open("src/utils/model_params.json", "r") as f:
            model_data = json.load(f)

        desired_model = model_name.value
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


@dataclass
class TrainingParams:
    LR_G = 0.003
    LR_C = 0.004
    LR_DECAY = 0.9
    LAMBDA_GP = 5
    CRITIC_STEPS = 5
    N_EPOCHS = 20

    @property
    def SHOW_GENERATED_INTERVAL(self) -> int:
        return int(self.N_EPOCHS / 4)

    @property
    def SAVE_INTERVAL(self) -> int:
        return int(self.N_EPOCHS / 1)


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
    def WINDOW(self) -> NDArray[np.float64]:
        return hann(self.FT_WIN)

    @property
    def FT_WIN(self) -> int:
        return (self.LINEAR_SPEC_FBINS - 1) * 2

    @property
    def FT_HOP(self) -> int:
        return int(self.sample_length * self.SR) // self.FRAMES
