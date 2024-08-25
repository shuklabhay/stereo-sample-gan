import torch
import torch.nn as nn
from utils.helpers import N_CHANNELS, N_FRAMES, N_FREQ_BINS
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Constants
BATCH_SIZE = 16
LATENT_DIM = 128


# Model Components
class LinearAttention(nn.Module):
    def __init__(self, in_channels):
        super(LinearAttention, self).__init__()
        self.reduced_channels = max(in_channels // 8, 1)
        self.query = nn.Conv2d(
            in_channels, self.reduced_channels, kernel_size=1, groups=1
        )
        self.key = nn.Conv2d(
            in_channels, self.reduced_channels, kernel_size=1, groups=1
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()

        query = self.query(x).view(batch, -1, height * width)
        key = self.key(x).view(batch, -1, height * width).permute(0, 2, 1)
        value = self.value(x).view(batch, -1, height * width)

        attention = torch.bmm(key, query)
        attention = F.normalize(F.relu(attention), p=1, dim=1)

        out = torch.bmm(value, attention)
        out = out.view(batch, channels, height, width)
        out = self.gamma * out + x
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # Shape: (BATCH_SIZE, 128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # Shape: (BATCH_SIZE, 64, 8, 8)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # LinearAttention(32),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # Shape: (BATCH_SIZE, 32, 16, 16)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # Shape: (BATCH_SIZE, 16, 32, 32)
            nn.ConvTranspose2d(16, 8, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),  # Shape: (BATCH_SIZE, 8, 128, 128)
            nn.ConvTranspose2d(8, N_CHANNELS, kernel_size=6, stride=2, padding=2),
            # Shape: (BATCH_SIZE, N_CHANNELS, 256, 256)
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
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            spectral_norm(nn.Conv2d(N_CHANNELS, 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8),
            spectral_norm(nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            LinearAttention(16),
            spectral_norm(nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            spectral_norm(nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0)),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def extract_features(self, x):
        feature_indices = [3, 9]  # Conv block index
        features = []
        for i, layer in enumerate(self.conv_blocks):
            x = layer(x)
            if i in feature_indices:
                features.append(x)
        return features

    def forward(self, x):
        x = self.conv_blocks(x)
        return x
