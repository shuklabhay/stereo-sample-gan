import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from utils.helpers import ModelParams, SignalConstants


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
            nn.ConvTranspose2d(
                ModelParams.LATENT_DIM, 128, kernel_size=4, stride=1, padding=0
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),  # Shape: (BATCH_SIZE, 128, 4, 4)
            nn.Dropout(ModelParams.DROPOUT_RATE),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),  # Shape: (BATCH_SIZE, 64, 8, 8)
            nn.Dropout(ModelParams.DROPOUT_RATE),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),  # Shape: (BATCH_SIZE, 32, 16, 16)
            nn.Dropout(ModelParams.DROPOUT_RATE),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),  # Shape: (BATCH_SIZE, 16, 32, 32)
            nn.Dropout(ModelParams.DROPOUT_RATE),
            nn.ConvTranspose2d(16, 8, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),  # Shape: (BATCH_SIZE, 8, 128, 128)
            nn.Dropout(ModelParams.DROPOUT_RATE),
            nn.ConvTranspose2d(
                8, SignalConstants.CHANNELS, kernel_size=6, stride=2, padding=2
            ),
            nn.Tanh(),  # Shape: (BATCH_SIZE, N_CHANNELS, 256, 256)
        )

    def forward(self, z):
        x = self.deconv_blocks(z)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv_blocks = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    SignalConstants.CHANNELS, 4, kernel_size=4, stride=2, padding=1
                )
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            LinearAttention(16),
            spectral_norm(nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Dropout(ModelParams.DROPOUT_RATE),
            spectral_norm(nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0)),
            nn.Flatten(),
        )

    def extract_features(self, x):
        """Extract features for x from specific layers."""
        features = [
            x
            for i, layer in enumerate(self.conv_blocks)
            if i == 0
            or isinstance(layer, LinearAttention)
            or i == len(self.conv_blocks) - 2
        ]

        return features

    def forward(self, x):
        x = self.conv_blocks(x)
        return x
