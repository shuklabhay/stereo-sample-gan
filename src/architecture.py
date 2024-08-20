import torch
import torch.nn as nn
from utils.helpers import N_CHANNELS, N_FRAMES, N_FREQ_BINS

# Constants Constants
BATCH_SIZE = 16
LATENT_DIM = 100
N_EPOCHS = 10

VALIDATION_INTERVAL = int(N_EPOCHS / 2)
SAVE_INTERVAL = int(N_EPOCHS / 1)


# Model Components
class PhaseShuffleLayer(nn.Module):
    def __init__(self, max_shift=2):
        super(PhaseShuffleLayer, self).__init__()
        self.max_shift = max_shift

    def forward(self, x):
        batch_size, channels, frames, freq = x.size()
        shifts = torch.randint(
            -self.max_shift,
            self.max_shift + 1,
            (batch_size, channels, 1, 1),
            device=x.device,
        )
        shifts = shifts.expand(-1, -1, frames, freq)

        idx = (
            torch.arange(frames, device=x.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        idx_shifted = idx + shifts
        idx_shifted = torch.clamp(idx_shifted, 0, frames - 1)

        return x.gather(2, idx_shifted)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, N_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Upsample(
                size=(N_FRAMES, N_FREQ_BINS), mode="bilinear", align_corners=False
            ),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.deconv_blocks(z)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Upsample(size=(512, 512), mode="bilinear", align_corners=False),
            nn.Conv2d(N_CHANNELS, 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(8),
            PhaseShuffleLayer(max_shift=2),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),
            PhaseShuffleLayer(max_shift=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            PhaseShuffleLayer(max_shift=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            PhaseShuffleLayer(max_shift=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            PhaseShuffleLayer(max_shift=2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            PhaseShuffleLayer(max_shift=2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),
            PhaseShuffleLayer(max_shift=2),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, 1)
        return x
