import torch
from architecture import LATENT_DIM, Critic, Generator
import numpy as np
from torch.optim.rmsprop import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from utils.file_helpers import (
    get_device,
    save_model,
)
from utils.signal_helpers import graph_spectrogram


# Training params
N_EPOCHS = 14
SHOW_GENERATED_INTERVAL = int(N_EPOCHS / 4)
SAVE_INTERVAL = int(N_EPOCHS / 1)

LR_G = 0.003
LR_C = 0.004
LAMBDA_GP = 5
CRITIC_STEPS = 5


# Loss metrics
def compute_g_loss(critic, fake_validity, fake_audio_data, real_audio_data):
    wasserstein_dist = -torch.mean(fake_validity)
    feat_match = 0.35 * calculate_feature_match_diff(
        critic, real_audio_data, fake_audio_data
    )

    computed_g_loss = (
        # generator total loss
        wasserstein_dist
        + feat_match
    )
    return computed_g_loss


def compute_c_loss(
    critic,
    fake_validity,
    real_validity,
    fake_audio_data,
    real_audio_data,
    training,
    device,
):
    wasserstein_dist = compute_wasserstein_diff(real_validity, fake_validity)
    spectral_diff = 0.15 * calculate_spectral_diff(real_audio_data, fake_audio_data)
    spectral_convergence = 0.15 * calculate_spectral_convergence_diff(
        real_audio_data, fake_audio_data
    )

    computed_c_loss = (
        # discrim total loss
        wasserstein_dist
        + spectral_diff
        + spectral_convergence
    )

    if training:
        gradient_penalty = calculate_gradient_penalty(
            critic, real_audio_data, fake_audio_data, device
        )
        computed_c_loss += LAMBDA_GP * gradient_penalty

    return computed_c_loss


# Loss Metrics
def compute_wasserstein_diff(real_validity, fake_validity):
    return -(torch.mean(real_validity) - torch.mean(fake_validity))


def calculate_feature_match_diff(critic, real_audio_data, fake_audio_data):
    real_features = critic.extract_features(real_audio_data)
    fake_features = critic.extract_features(fake_audio_data)

    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += torch.mean(torch.abs(real_feat - fake_feat))

    return loss / len(real_features)


def calculate_spectral_diff(real_audio_data, fake_audio_data):
    return torch.mean(torch.abs(real_audio_data - fake_audio_data))


def calculate_spectral_convergence_diff(real_audio_data, fake_audio_data):
    numerator = torch.norm(fake_audio_data - real_audio_data, p=2)
    denominator = torch.norm(real_audio_data, p=2) + 1e-8
    return numerator / denominator


def calculate_gradient_penalty(critic, real_samples, fake_samples, device):
    real_samples.requires_grad_(True)
    fake_samples.requires_grad_(True)

    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(
        True
    )
    c_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Training
def train_epoch(
    generator,
    critic,
    dataloader,
    optimizer_G,
    optimizer_C,
    scheduler_G,
    scheduler_C,
    device,
    epoch_number,
):
    generator.train()
    critic.train()
    total_g_loss, total_c_loss, total_w_dist = 0, 0, 0

    for i, (real_audio_data,) in enumerate(dataloader):
        batch = real_audio_data.size(0)
        real_audio_data = real_audio_data.to(device)

        # Train critic
        optimizer_C.zero_grad()
        z = torch.randn(batch, LATENT_DIM, 1, 1).to(device)
        fake_audio_data = generator(z)
        real_validity = critic(real_audio_data)
        fake_validity = critic(fake_audio_data.detach())

        c_loss = compute_c_loss(
            critic,
            fake_validity,
            real_validity,
            fake_audio_data,
            real_audio_data,
            True,
            device,
        )
        c_loss.backward()
        optimizer_C.step()

        total_c_loss += c_loss.item()
        total_w_dist += compute_wasserstein_diff(real_validity, fake_validity).item()

        # Train generator every CRITIC_STEPS steps
        if i % CRITIC_STEPS == 0:
            optimizer_G.zero_grad()
            fake_audio_data = generator(z)
            fake_validity = critic(fake_audio_data)

            g_loss = compute_g_loss(
                critic, fake_validity, fake_audio_data, real_audio_data
            )
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()

            # # Save training progress images
            # if i % (CRITIC_STEPS * 14) == 0:
            #     fake_audio_to_visualize = fake_audio_data[0].cpu().detach().numpy()
            #     graph_spectrogram(
            #         fake_audio_to_visualize,
            #         f"diverse_generator_epoch_{epoch_number + 1}_step_{i}.png",
            #         True,
            #     )

    avg_g_loss = total_g_loss / len(dataloader)
    avg_c_loss = total_c_loss / len(dataloader)
    avg_w_dist = total_w_dist / len(dataloader)

    scheduler_G.step()
    scheduler_C.step()

    return avg_g_loss, avg_c_loss, avg_w_dist


def validate(generator, critic, dataloader, device):
    generator.eval()
    critic.eval()
    total_g_loss, total_c_loss, total_w_dist = 0, 0, 0

    with torch.no_grad():
        for (real_audio_data,) in dataloader:
            batch = real_audio_data.size(0)
            real_audio_data = real_audio_data.to(device)

            z = torch.randn(batch, LATENT_DIM, 1, 1).to(device)
            fake_audio_data = generator(z)

            real_validity = critic(real_audio_data)
            fake_validity = critic(fake_audio_data)

            g_loss = compute_g_loss(
                critic, fake_validity, fake_audio_data, real_audio_data
            )
            c_loss = compute_c_loss(
                critic,
                fake_validity,
                real_validity,
                fake_audio_data,
                real_audio_data,
                False,
                device,
            )

            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()
            total_w_dist += compute_wasserstein_diff(
                real_validity, fake_validity
            ).item()

    return (
        total_g_loss / len(dataloader),
        total_c_loss / len(dataloader),
        total_w_dist / len(dataloader),
    )


def training_loop(train_loader, val_loader):
    # Initialize models and optimizers
    generator = Generator()
    critic = Critic()
    optimizer_G = RMSprop(generator.parameters(), lr=LR_G, weight_decay=0.05)
    optimizer_C = RMSprop(critic.parameters(), lr=LR_C, weight_decay=0.05)

    LR_DECAY = 0.9
    scheduler_G = ExponentialLR(optimizer_G, gamma=LR_DECAY)
    scheduler_C = ExponentialLR(optimizer_C, gamma=LR_DECAY)

    # Train
    device = get_device()
    generator.to(device)
    critic.to(device)

    best_val_w_dist = float("inf")  # Initialize
    epochs_no_improve = 0
    patience = 3  # epochs
    warmup = 4  # epochs
    for epoch in range(N_EPOCHS):
        # Train
        train_g_loss, train_c_loss, train_w_dist = train_epoch(
            generator,
            critic,
            train_loader,
            optimizer_G,
            optimizer_C,
            scheduler_G,
            scheduler_C,
            device,
            epoch,
        )
        print(
            f"[{epoch+1}/{N_EPOCHS}] Train - G Loss: {train_g_loss:.6f}, C Loss: {train_c_loss:.6f}, W Dist: {train_w_dist:.6f}"
        )

        # Validate
        val_g_loss, val_c_loss, val_w_dist = validate(
            generator, critic, val_loader, device
        )
        print(
            f"------ Val ------ G Loss: {val_g_loss:.6f}, C Loss: {val_c_loss:.6f}, W Dist: {val_w_dist:.6f}"
        )

        # Generate example audio
        if (epoch + 1) % SHOW_GENERATED_INTERVAL == 0:
            examples_to_generate = 3
            z = torch.randn(examples_to_generate, LATENT_DIM, 1, 1).to(device)
            generated_audio = generator(z).squeeze()

            for i in range(examples_to_generate):
                generated_audio_np = generated_audio[i].cpu().detach().numpy()
                graph_spectrogram(
                    generated_audio_np,
                    f"Epoch {epoch + 1} Generated Audio #{i + 1}",
                )

        # Early exit/saving
        if (epoch + 1) >= warmup:
            save_model(generator)
            if np.abs(val_w_dist) < best_val_w_dist:
                best_val_w_dist = np.abs(val_w_dist)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"epochs without w_dist improvement: {epochs_no_improve}")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered")
                    break
