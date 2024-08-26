import numpy as np
import os
import torch

# Constants
audio_data_dir = "data/kick_samples"
sinetest_data_dir = "data/sine_test"
compiled_data_path = "data/compiled_data.npy"
average_spectrogram_path = "data/average_spectrogram.npy"
audio_output_dir = "model"
model_save_dir = "model"


# File Utility
def load_npy_data(file_path):
    return np.load(file_path, allow_pickle=True)


def save_model(model, name, preserve_old=False):
    # Clear previous models
    if preserve_old is not True:
        for filename in os.listdir(model_save_dir):
            file_path = os.path.join(model_save_dir, filename)
            os.remove(file_path)

    # Save model
    torch.save(
        model.state_dict(),
        f"{model_save_dir}/{name}.pth",
    )
    print(f"Model Saved")


def get_device():
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    # else:
    #     return torch.device("cpu")
    return torch.device("cpu")


def check_and_delete_DSStore(current_directory):
    DSStore_path = os.path.join(current_directory, ".DS_Store")
    if os.path.exists(DSStore_path):
        os.remove(DSStore_path)


def save_freq_info(freq_info, save_path):
    np.save(save_path, freq_info)
