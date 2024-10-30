import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from train import training_loop
from utils.helpers import (
    DataUtils,
    ModelParams,
    ModelUtils,
    SignalProcessing,
    TrainingParams,
)

# Load params
model_params = ModelParams()
utils = DataUtils()
training_params = TrainingParams()
model_utils = ModelUtils(model_params.sample_length)
signal_processing = SignalProcessing(model_params.sample_length)

# Load data
print("Encoding audio data for", model_params.selected_model)
signal_processing.encode_sample_directory(
    model_params.training_audio_dir, model_params.compiled_data_path, False
)


all_spectrograms = utils.load_loudness_data(model_params.compiled_data_path)
all_spectrograms = torch.FloatTensor(all_spectrograms)
print("Data encoded:", all_spectrograms.shape)

train_size = int(0.8 * len(all_spectrograms))
val_size = len(all_spectrograms) - train_size
train_dataset, val_dataset = random_split(
    TensorDataset(all_spectrograms), [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset, batch_size=model_params.BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=model_params.BATCH_SIZE, shuffle=False)


# Train
print("Starting training for", model_params.selected_model)
training_loop(train_loader, val_loader)
