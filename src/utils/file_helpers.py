import numpy as np
import os
import torch
import soundfile as sf

from usage_params import model_save_path

# Constants
GLOBAL_SR = 44100


# File Utility
def load_loudness_data(file_path):
    return np.load(file_path, allow_pickle=True)


def save_loudness_data(loudness_information, save_path):
    np.save(save_path, loudness_information)


def save_audio(save_path, audio):
    sf.write(save_path, audio.T, GLOBAL_SR)


def save_model(model, name):
    # Save model
    torch.save(
        model.state_dict(),
        model_save_path,
    )
    print(f"Model Saved")


def get_device():
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    # else:
    #     return torch.device("cpu")
    return torch.device("cpu")  # personal hardware limitations


def delete_DSStore(current_directory):
    DSStore_path = os.path.join(current_directory, ".DS_Store")
    if os.path.exists(DSStore_path):
        os.remove(DSStore_path)
