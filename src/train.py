import torch
import torch.nn.functional as F
from architecture import LATENT_DIM
from utils.helpers import (
    average_spectrogram_path,
    graph_spectrogram,
    load_npy_data,
    save_model,
    scale_data_to_range,
)

# Constants
N_EPOCHS = 10
VALIDATION_INTERVAL = int(N_EPOCHS / 2)
SAVE_INTERVAL = int(N_EPOCHS / 1)


# Helpers
def calculate_weighted_spectrogram_decay_penalty(
    generated_audio, spectrogram_bank, device
):
    # Load average & random spectrogram batches
    batch = generated_audio.shape[0]
    average_spectrogram = load_npy_data(average_spectrogram_path)
    average_spectrogram = torch.from_numpy(average_spectrogram).to(device)
    average_spectrogram_batch = average_spectrogram.repeat(batch, 1, 1, 1)

    random_indices = torch.randint(0, len(spectrogram_bank), (batch,), device=device)
    random_training_batch = spectrogram_bank[random_indices]

    # Created weighted reference based on random sample & avg sample similarity
    similarity_to_average = 1 - F.cosine_similarity(
        random_training_batch.view(batch, -1), average_spectrogram_batch.view(batch, -1)
    )  # further from avg = larger weight for random
    weights = F.softmax(similarity_to_average, dim=0).view(batch, 1, 1, 1)
    weighted_reference = (
        weights * random_training_batch + (1 - weights) * average_spectrogram_batch
    )

    # Compare generated audio to reference
    decay_penalty = F.mse_loss(generated_audio, weighted_reference)
    return decay_penalty


def calculate_energy_decay_penalty(generated_audio):
    energy_over_time = torch.sum(generated_audio**2, dim=(1, 3))

    energy_diff = energy_over_time[:, 1:] - energy_over_time[:, :-1]
    decay_penalty = torch.mean(F.relu(energy_diff))
    return decay_penalty


def calculate_feat_match_penalty(real_features, fake_features):
    loss = 0
    for r_feat, f_feat in zip(real_features, fake_features):
        loss += F.mse_loss(r_feat.mean(0), f_feat.mean(0))

    return loss


def calculate_kick_character_penalty(generated_audio, device):
    batch_size, channels, time_steps, freq_bins = generated_audio.shape
    energy_over_time = torch.sum(generated_audio**2, dim=(1, 3))

    # encourage transient
    transient_penalty = -torch.mean(energy_over_time[:, :10])

    # encourage decay
    energy_diff = energy_over_time[:, 1:] - energy_over_time[:, :-1]
    decay_penalty = torch.mean(F.relu(energy_diff))

    # encourage high to low freq transition
    freq_range = torch.arange(freq_bins, device=device).unsqueeze(0).unsqueeze(0)
    spectral_centroid = torch.sum(
        torch.abs(generated_audio) * freq_range, dim=(1, 3)
    ) / (torch.sum(torch.abs(generated_audio), dim=(1, 3)) + 1e-8)
    centroid_shift = torch.mean(spectral_centroid[:, 1:] - spectral_centroid[:, :-1])
    centroid_shift_penalty = F.relu(-centroid_shift)

    # encourage different lengths
    duration = torch.argmin(F.relu(0.1 - energy_over_time), dim=1)
    duration_variance = torch.var(duration.float())
    duration_penalty = -torch.log(duration_variance + 1e-6)

    total_penalty = (
        transient_penalty + decay_penalty + centroid_shift_penalty + duration_penalty
    )

    return total_penalty


def calculate_diversity_penalty(generated_samples):
    batch_size = generated_samples.size(0)
    flattened = generated_samples.view(batch_size, -1)

    similarity_matrix = F.cosine_similarity(
        flattened.unsqueeze(1), flattened.unsqueeze(0), dim=2
    )
    mean_similarity = (
        torch.sum(similarity_matrix) - torch.trace(similarity_matrix)
    ) / (batch_size * (batch_size - 1))

    diversity_penalty = -mean_similarity

    return diversity_penalty


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

        def smooth_labels(tensor, amount=0.1):
            return tensor + amount * torch.rand_like(tensor)

        real_labels = smooth_labels(torch.ones(batch_size, 1).to(device))
        fake_labels = smooth_labels(torch.zeros(batch_size, 1).to(device))

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
        fake_audio_data = generator(z)

        # Loss calculations
        g_adv_loss = criterion(discriminator(fake_audio_data), real_labels)

        feat_match_weight = 0.1
        # real_features = discriminator.get_features(real_audio_data)
        # fake_features = discriminator.get_features(fake_audio_data)
        # feat_match_penalty = calculate_feat_match_penalty(real_features, fake_features)

        kick_character_penalty_weight = 0.1
        kick_character_penalty = calculate_kick_character_penalty(
            fake_audio_data, device
        )

        diversity_penalty_weight = 0.1
        diversity_penalty = calculate_diversity_penalty(fake_audio_data)

        print(g_adv_loss, kick_character_penalty, diversity_penalty)
        # Combine losses
        g_loss = (
            g_adv_loss
            # + feat_match_penalty * feat_match_weight
            + kick_character_penalty * kick_character_penalty_weight
            + diversity_penalty * diversity_penalty_weight
        )

        g_loss.backward()
        optimizer_G.step()
        total_g_loss += g_loss.item()

        # Train discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_audio_data), real_labels)
        fake_loss = criterion(discriminator(fake_audio_data.detach()), fake_labels)

        d_loss = (real_loss + fake_loss) / 2
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

            z = torch.randn(1, LATENT_DIM, 1, 1).to(device)
            generated_audio = generator(z).squeeze()
            generated_audio_np = generated_audio.cpu().detach().numpy()
            graph_spectrogram(
                scale_data_to_range(generated_audio_np, -120, 40),
                f"Generated Audio epoch {epoch+1}",
            )

        # Save models periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_model(generator, "DCGAN")
