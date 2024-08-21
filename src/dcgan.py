import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from architecture import (
    BATCH_SIZE,
    Discriminator,
    Generator,
)
from train import training_loop
from utils.helpers import (
    compiled_data_path,
    get_device,
    load_npy_data,
)

# Constants
LR_G = 0.002
LR_D = 0.001

# Load data
audio_data = load_npy_data(compiled_data_path)
audio_data = torch.FloatTensor(audio_data)
train_size = int(0.8 * len(audio_data))
val_size = len(audio_data) - train_size
train_dataset, val_dataset = random_split(
    TensorDataset(audio_data), [train_size, val_size]
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize models and optimizers
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))  # type: ignore
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))  # type: ignore

device = get_device()
generator.to(device)
discriminator.to(device)


training_loop(
    generator,
    discriminator,
    train_loader,
    val_loader,
    criterion,
    optimizer_G,
    optimizer_D,
    device,
)
