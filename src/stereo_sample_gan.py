import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from architecture import (
    BATCH_SIZE,
)
from train import training_loop
from utils.file_helpers import (
    get_device,
    load_loudness_data,
)

from data_processing.encode_audio_data import compiled_data_path

# Constants
LR_G = 0.003
LR_C = 0.004

# Load data
all_spectrograms = load_loudness_data(compiled_data_path)
all_spectrograms = torch.FloatTensor(all_spectrograms)
train_size = int(0.8 * len(all_spectrograms))
val_size = len(all_spectrograms) - train_size
train_dataset, val_dataset = random_split(
    TensorDataset(all_spectrograms), [train_size, val_size]
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Train
training_loop(train_loader, val_loader)
