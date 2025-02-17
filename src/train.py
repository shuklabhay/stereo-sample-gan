from typing import Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader

from architecture import Critic, Generator
from utils.helpers import DataUtils, ModelParams, ModelUtils, TrainingParams

# Initialize parameters
model_params = ModelParams()
training_params = TrainingParams()
model_utils = ModelUtils(model_params.sample_length)


def compute_g_loss(
    critic: Critic,
    fake_validity: torch.Tensor,
    fake_audio_data: torch.Tensor,
    real_audio_data: torch.Tensor,
) -> torch.Tensor:
    """Calculate generator loss."""
    wasserstein_dist = -torch.mean(fake_validity)
    feat_match = 0.45 * calculate_feature_match_diff(
        critic, real_audio_data, fake_audio_data
    )

    computed_g_loss = wasserstein_dist + feat_match
    return computed_g_loss


def compute_c_loss(
    critic: Critic,
    fake_validity: torch.Tensor,
    real_validity: torch.Tensor,
    fake_audio_data: torch.Tensor,
    real_audio_data: torch.Tensor,
    training: bool,
    device: torch.device,
) -> torch.Tensor:
    """Calculate critic loss."""
    wasserstein_dist = calculate_wasserstein_diff(real_validity, fake_validity)
    spectral_diff = 0.15 * calculate_spectral_diff(real_audio_data, fake_audio_data)
    spectral_convergence = 0.15 * calculate_spectral_convergence_diff(
        real_audio_data, fake_audio_data
    )

    computed_c_loss = wasserstein_dist + spectral_diff + spectral_convergence

    if training:
        gradient_penalty = calculate_gradient_penalty(
            critic, real_audio_data, fake_audio_data, device
        )
        computed_c_loss += training_params.LAMBDA_GP * gradient_penalty

    return computed_c_loss


def calculate_wasserstein_diff(
    real_validity: torch.Tensor, fake_validity: torch.Tensor
) -> torch.Tensor:
    """Calculate wasserstien loss metric."""
    return -(torch.mean(real_validity) - torch.mean(fake_validity))


def calculate_feature_match_diff(
    critic: Critic, real_audio_data: torch.Tensor, fake_audio_data: torch.Tensor
) -> torch.Tensor:
    """Calculate feature match difference loss metric."""
    real_features = critic.extract_features(real_audio_data)
    fake_features = critic.extract_features(fake_audio_data)

    loss = torch.tensor(0.0, device=real_audio_data.device)
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += torch.mean(torch.abs(real_feat - fake_feat))

    return loss / len(real_features)


def calculate_spectral_diff(
    real_audio_data: torch.Tensor, fake_audio_data: torch.Tensor
) -> torch.Tensor:
    """Calculate spectral difference loss metric."""
    return torch.mean(torch.abs(real_audio_data - fake_audio_data))


def calculate_spectral_convergence_diff(
    real_audio_data: torch.Tensor, fake_audio_data: torch.Tensor
) -> torch.Tensor:
    """Calculate spectral convergence loss metric."""
    numerator = torch.norm(fake_audio_data - real_audio_data, p=2)
    denominator = torch.norm(real_audio_data, p=2) + 1e-8
    return numerator / denominator


def calculate_gradient_penalty(
    critic: Critic,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Calculate gradient penalty loss metric."""
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


def train_epoch(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    optimizer_G: RMSprop,
    optimizer_C: RMSprop,
    scheduler_G: ExponentialLR,
    scheduler_C: ExponentialLR,
    device: torch.device,
    epoch_number: int,
) -> Tuple[float, float, float]:
    """Training."""
    generator.train()
    critic.train()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0

    for i, (real_audio_data,) in enumerate(dataloader):
        batch = real_audio_data.size(0)
        real_audio_data = real_audio_data.to(device)

        # Train critic
        optimizer_C.zero_grad()
        z = torch.randn(batch, model_params.LATENT_DIM, 1, 1).to(device)
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
        total_w_dist += calculate_wasserstein_diff(real_validity, fake_validity).item()

        # Train generator every CRITIC_STEPS steps
        if i % training_params.CRITIC_STEPS == 0:
            optimizer_G.zero_grad()
            fake_audio_data = generator(z)
            fake_validity = critic(fake_audio_data)

            g_loss = compute_g_loss(
                critic, fake_validity, fake_audio_data, real_audio_data
            )
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()

    avg_g_loss = total_g_loss / len(dataloader)
    avg_c_loss = total_c_loss / len(dataloader)
    avg_w_dist = total_w_dist / len(dataloader)

    scheduler_G.step()
    scheduler_C.step()

    return avg_g_loss, avg_c_loss, avg_w_dist


def validate(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validation."""
    generator.eval()
    critic.eval()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0

    with torch.no_grad():
        for (real_audio_data,) in dataloader:
            batch = real_audio_data.size(0)
            real_audio_data = real_audio_data.to(device)

            z = torch.randn(batch, model_params.LATENT_DIM, 1, 1).to(device)
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
            total_w_dist += calculate_wasserstein_diff(
                real_validity, fake_validity
            ).item()

    return (
        total_g_loss / len(dataloader),
        total_c_loss / len(dataloader),
        total_w_dist / len(dataloader),
    )


def training_loop(train_loader: DataLoader, val_loader: DataLoader) -> None:
    """Training loop."""
    generator = Generator()
    critic = Critic()
    optimizer_G = RMSprop(
        generator.parameters(), lr=training_params.LR_G, weight_decay=0.05
    )
    optimizer_C = RMSprop(
        critic.parameters(), lr=training_params.LR_C, weight_decay=0.05
    )

    scheduler_G = ExponentialLR(optimizer_G, gamma=training_params.LR_DECAY)
    scheduler_C = ExponentialLR(optimizer_C, gamma=training_params.LR_DECAY)

    device = model_utils.get_device()
    generator.to(device)
    critic.to(device)

    best_val_w_dist = float("inf")
    epochs_no_improve = 0
    patience = 3
    warmup = 4

    for epoch in range(training_params.N_EPOCHS):
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
            f"[{epoch+1}/{training_params.N_EPOCHS}] Train - G Loss: {train_g_loss:.6f}, C Loss: {train_c_loss:.6f}, W Dist: {train_w_dist:.6f}"
        )

        # Validate
        val_g_loss, val_c_loss, val_w_dist = validate(
            generator, critic, val_loader, device
        )
        print(
            f"------ Val ------ G Loss: {val_g_loss:.6f}, C Loss: {val_c_loss:.6f}, W Dist: {val_w_dist:.6f}"
        )

        # Generate example audio
        if (epoch + 1) % training_params.SHOW_GENERATED_INTERVAL == 0:
            examples_to_generate = 3
            z = torch.randn(examples_to_generate, model_params.LATENT_DIM, 1, 1).to(
                device
            )
            generated_audio = generator(z).squeeze()

            for i in range(examples_to_generate):
                generated_audio_np = generated_audio[i].cpu().detach().numpy()
                DataUtils.graph_spectrogram(
                    generated_audio_np,
                    f"Epoch {epoch + 1} Generated Audio #{i + 1}",
                )

        # Early exit/saving
        if (epoch + 1) >= warmup:
            model_utils.save_model(generator)
            if np.abs(val_w_dist) < best_val_w_dist:
                best_val_w_dist = np.abs(val_w_dist)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"epochs without w_dist improvement: {epochs_no_improve}")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered")
                    break
