import torch
import torch.nn.functional as F
from architecture import LATENT_DIM
from utils.helpers import (
    graph_spectrogram,
    save_model,
    scale_data_to_range,
)

torch.autograd.set_detect_anomaly(True)

# Constants
N_EPOCHS = 1
VALIDATION_INTERVAL = 1
SAVE_INTERVAL = int(N_EPOCHS / 1)


# Utility functions
def smooth_labels(tensor, isZeros):
    amount = 0.05

    if isZeros is True:
        return tensor + amount * torch.rand_like(tensor)
    else:
        return tensor - amount * torch.rand_like(tensor)


# Generator Loss Metrics
def calculate_feat_match_penalty(real_features, fake_features):
    loss = 0
    for r_feat, f_feat in zip(real_features, fake_features):
        loss += F.mse_loss(r_feat.mean(0), f_feat.mean(0))

    return loss


def calculate_spectral_centroid_shift_penalty(
    generated_audio, training_audio_data, device
):
    batch_size, channels, frames, freq_bins = generated_audio.shape
    transient_frames = 5
    decay_frames = 150
    release_frames = frames - decay_frames - transient_frames

    # Calculate spectral centroid
    freq_range = torch.arange(freq_bins, device=device).unsqueeze(0).unsqueeze(0)
    spectral_centroid = torch.sum(
        torch.abs(generated_audio) * freq_range, dim=(1, 3)
    ) / (torch.sum(torch.abs(generated_audio), dim=(1, 3)) + 1e-8)

    # Transient section: shift downwards
    transient_centroid_shift = torch.mean(
        spectral_centroid[:, 1:transient_frames]
        - spectral_centroid[:, : transient_frames - 1]
    )
    transient_penalty = F.relu(transient_centroid_shift)

    # Decay section: shift downwards
    decay_centroid = spectral_centroid[
        :, transient_frames : transient_frames + decay_frames
    ]
    decay_trend = torch.mean(decay_centroid[:, 1:] - decay_centroid[:, :-1], dim=1)
    decay_penalty = F.relu(torch.mean(decay_trend))

    # Encourage diversity in decay rates
    decay_rates = torch.diff(decay_centroid, dim=1)
    decay_rate_variance = torch.var(decay_rates, dim=1)
    decay_diversity_bonus = -torch.mean(decay_rate_variance)

    # Release section: shift downwards
    release_centroid_shift = torch.mean(
        spectral_centroid[:, -release_frames:]
        - spectral_centroid[:, -release_frames - 1 : -1]
    )
    release_penalty = F.relu(release_centroid_shift)

    # Calculate total penalty
    total_penalty = (
        transient_penalty
        + decay_penalty
        # + 0.0005 * decay_diversity_bonus
        + release_penalty
    )

    # print(
    #     f"Transient Loss: {transient_penalty.item():.4f}, Decay Loss: {decay_penalty.item():.4f}, "
    #     f"Decay Div Loss: {decay_diversity_bonus.item():.4f}, Release Loss: {release_penalty.item():.4f}, "
    #     f"Total Loss: {total_penalty.item():.4f}"
    # )

    return total_penalty


def calculate_diversity_penalty(generated_audio, training_audio_data, device):
    # Encourage global sample diversity
    batch_size = generated_audio.size(0)
    random_indices = torch.randint(
        0, len(training_audio_data), (batch_size,), device=device
    )
    sampled_training_audio = training_audio_data[random_indices]

    similarity = F.cosine_similarity(
        generated_audio.view(batch_size, -1),
        sampled_training_audio.view(batch_size, -1),
    )

    diversity_penalty = torch.mean(similarity)

    return diversity_penalty


# Discriminator Loss Metrics
def calculate_spectral_diff(real_audio_data, fake_audio_data):
    spectral_diff = torch.mean(torch.abs(real_audio_data - fake_audio_data))

    return torch.mean(spectral_diff)


# Training
def train_epoch(
    generator,
    discriminator,
    dataloader,
    criterion,
    optimizer_G,
    optimizer_D,
    training_audio_data,
    device,
):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0, 0

    for i, (real_audio_data,) in enumerate(dataloader):
        batch_size = real_audio_data.size(0)
        real_audio_data = real_audio_data.to(device)

        real_labels = smooth_labels(
            (torch.ones(batch_size).unsqueeze(1)).to(device), False
        )
        fake_labels = smooth_labels(
            (torch.zeros(batch_size).unsqueeze(1)).to(device), True
        )

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
        fake_audio_data = generator(z)

        # Generator Loss
        g_adv_loss = criterion(discriminator(fake_audio_data), real_labels)

        real_features = discriminator.get_features(real_audio_data)
        fake_features = discriminator.get_features(fake_audio_data)
        feat_match_penalty = 0.5 * calculate_feat_match_penalty(
            real_features, fake_features
        )

        centroid_shift_penalty = 0.3 * calculate_spectral_centroid_shift_penalty(
            fake_audio_data, training_audio_data, device
        )
        diversity_penalty = 0.4 * calculate_diversity_penalty(
            fake_audio_data, training_audio_data, device
        )  # samples in a batch unique

        # MAKE FEAT MATCH PENALTY WORK WITH ONLY REAL SMAPLES NOT WHATEVER DISCRIM TRAINING ON

        g_loss = (
            # force vertical
            g_adv_loss
            # + feat_match_penalty
            # + centroid_shift_penalty
            # + diversity_penalty
        )

        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        total_g_loss += g_loss.item()

        # Train discriminator
        optimizer_D.zero_grad()
        real_audio_data_noisy = discriminator.add_noise(real_audio_data)
        fake_audio_data_noisy = discriminator.add_noise(fake_audio_data.detach())

        real_loss = criterion(discriminator(real_audio_data_noisy), real_labels)
        fake_loss = criterion(discriminator(fake_audio_data_noisy), fake_labels)

        # Discriminator Loss
        d_adv_loss = (real_loss + fake_loss) / 2
        spectral_diff = 0.2 * calculate_spectral_diff(real_audio_data, fake_audio_data)

        d_loss = d_adv_loss + spectral_diff

        d_loss.backward()
        optimizer_D.step()
        total_d_loss += d_loss.item()

    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)


def validate(generator, discriminator, dataloader, criterion, device):
    generator.eval()
    discriminator.eval()
    total_g_loss, total_d_loss = 0, 0

    with torch.no_grad():
        for (real_audio_data,) in dataloader:
            batch_size = real_audio_data.size(0)
            real_audio_data = real_audio_data.to(device)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            z = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
            fake_audio_data = generator(z)

            g_loss = criterion(discriminator(fake_audio_data), real_labels)
            total_g_loss += g_loss.item()
            real_loss = criterion(discriminator(real_audio_data), real_labels)
            fake_loss = criterion(discriminator(fake_audio_data), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            total_d_loss += d_loss.item()

    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)


def training_loop(
    generator,
    discriminator,
    train_loader,
    val_loader,
    criterion,
    optimizer_G,
    optimizer_D,
    training_audio_data,
    device,
):
    for epoch in range(N_EPOCHS):
        train_g_loss, train_d_loss = train_epoch(
            generator,
            discriminator,
            train_loader,
            criterion,
            optimizer_G,
            optimizer_D,
            training_audio_data,
            device,
        )

        print(
            f"[{epoch+1}/{N_EPOCHS}] Train - G Loss: {train_g_loss:.6f}, D Loss: {train_d_loss:.6f}"
        )

        # Validate periodically
        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            val_g_loss, val_d_loss = validate(
                generator, discriminator, val_loader, criterion, device
            )
            print(
                f"------ Val ------ G Loss: {val_g_loss:.6f}, D Loss: {val_d_loss:.6f}"
            )

            examples_to_generate = 3
            z = torch.randn(examples_to_generate, LATENT_DIM, 1, 1).to(device)
            generated_audio = generator(z).squeeze()
            for i in range(examples_to_generate):
                generated_audio_np = generated_audio[i].cpu().detach().numpy()
                graph_spectrogram(
                    scale_data_to_range(generated_audio_np, -120, 40),
                    f"Generated Audio {i + 1} epoch {epoch + 1}",
                )

        # Save models periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_model(generator, "DCGAN")
