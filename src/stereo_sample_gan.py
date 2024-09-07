import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

from architecture import (
    BATCH_SIZE,
    Discriminator,
    Generator,
)
from train import training_loop
from utils.file_helpers import (
    get_device,
    load_loudness_data,
    compiled_data_path,
)

# Constants
LR_G = 0.004
LR_D = 0.005

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

# Initialize models and optimizers
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCEWithLogitsLoss()
optimizer_G = AdamW(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_D = AdamW(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))


# Train
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
