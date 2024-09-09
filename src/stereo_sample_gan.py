import torch
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader, TensorDataset, random_split

from architecture import (
    BATCH_SIZE,
    Critic,
    Generator,
)
from train import training_loop
from utils.file_helpers import (
    get_device,
    load_loudness_data,
    compiled_data_path,
)

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

# Initialize models and optimizers
generator = Generator()
critic = Critic()
optimizer_G = RMSprop(generator.parameters(), lr=LR_G, weight_decay=0.05)
optimizer_C = RMSprop(critic.parameters(), lr=LR_C, weight_decay=0.05)


# Train
device = get_device()
generator.to(device)
critic.to(device)
training_loop(
    generator,
    critic,
    train_loader,
    val_loader,
    optimizer_G,
    optimizer_C,
    device,
)
