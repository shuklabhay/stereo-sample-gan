import torch
import torch.nn.functional as F
from architecture import Critic, Generator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.constants import model_selection
from utils.evaluation import calculate_audio_metrics
from utils.helpers import DataUtils, ModelParams, ModelUtils, SignalProcessing

# Global initialization
model_params = ModelParams()
model_utils = ModelUtils(model_params.sample_length)
signal_processing = SignalProcessing(model_params.sample_length)


def compute_generator_loss(
    generated_validity: torch.Tensor,
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
) -> torch.Tensor:
    """Compute total generator loss."""
    adversarial_loss = -torch.mean(generated_validity)
    freq_loss = compute_frequency_loss(generated_specs, real_specs)
    decay_loss = compute_decay_loss(generated_specs, real_specs)
    spectral_loss = compute_multiscale_spectral_loss(generated_specs, real_specs)
    stereo_loss = compute_stereo_coherence_loss(generated_specs, real_specs)

    return (
        adversarial_loss
        + 0.6 * freq_loss
        + 0.2 * decay_loss
        + 0.3 * spectral_loss
        + 0.1 * stereo_loss
    )


def compute_frequency_loss(
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
) -> torch.Tensor:
    """Calculate frequency-domain loss between spectrograms."""
    generated_freq_energy = torch.mean(generated_specs, dim=3)
    real_freq_energy = torch.mean(real_specs, dim=3)
    return F.l1_loss(generated_freq_energy, real_freq_energy)


def compute_decay_loss(
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
) -> torch.Tensor:
    """Compute loss to follow original spectral decay pattern."""
    # Calculate mean energy per frame band
    generated_env = generated_specs.sum(dim=-1)
    real_env = real_specs.sum(dim=-1)

    # Compute temporal differences
    generated_diff = generated_env[:, :, 1:] - generated_env[:, :, :-1]
    real_diff = real_env[:, :, 1:] - real_env[:, :, :-1]

    # Find time steps of decay
    decay_mask = (real_diff < 0).float()

    # MSE between decay regions
    decay_loss_val = torch.sum(((generated_diff - real_diff) ** 2) * decay_mask) / (
        torch.sum(decay_mask) + 1e-8
    )

    total_loss = decay_loss_val
    return total_loss


def compute_multiscale_spectral_loss(
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
) -> torch.Tensor:
    """Compute L1 loss between spectrograms at multiple scales."""
    scales = [1, 2, 4]
    total_loss = 0.0
    for scale in scales:
        if scale > 1:
            gen_scaled = torch.nn.functional.avg_pool2d(
                generated_specs, kernel_size=scale, stride=scale
            )
            real_scaled = torch.nn.functional.avg_pool2d(
                real_specs, kernel_size=scale, stride=scale
            )
        else:
            gen_scaled = generated_specs
            real_scaled = real_specs
        total_loss += torch.nn.functional.l1_loss(gen_scaled, real_scaled)
    return total_loss / len(scales)


def compute_stereo_coherence_loss(
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
    lowband_cutoff: int = 30,
) -> torch.Tensor:
    """Compute stereo coherence loss across frequency bands."""
    # Split into left/right channels
    left_gen, right_gen = generated_specs[:, 0], generated_specs[:, 1]
    left_real, right_real = real_specs[:, 0], real_specs[:, 1]

    # Compute stereo difference tensors
    gen_diff = left_gen - right_gen
    real_diff = left_real - right_real

    # Enford mono lowend
    lowband_loss = torch.mean(torch.abs(gen_diff[..., :lowband_cutoff, :]))

    # Match high band stereo spread
    highband_diff = torch.abs(gen_diff[..., lowband_cutoff:, :])
    target_diff = torch.abs(real_diff[..., lowband_cutoff:, :])
    highband_loss = F.l1_loss(highband_diff, target_diff)

    return lowband_loss + highband_loss


def compute_critic_loss(
    critic: Critic,
    generated_validity: torch.Tensor,
    real_validity: torch.Tensor,
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Compute total critic loss."""
    wasserstein_loss = compute_wasserstein_loss(generated_validity, real_validity)
    spectral_loss = compute_spectral_loss(generated_specs, real_specs)
    convergence_loss = compute_spectral_convergence_loss(generated_specs, real_specs)

    loss = wasserstein_loss + 0.3 * spectral_loss + 0.3 * convergence_loss

    if training:
        gradient_penalty = compute_gradient_penalty(critic, real_specs, generated_specs)
        loss += model_params.LAMBDA_GP * gradient_penalty
    return loss


def compute_wasserstein_loss(
    generated_validity: torch.Tensor,
    real_validity: torch.Tensor,
) -> torch.Tensor:
    """Calculate Wasserstein distance between real and generated samples."""
    return torch.mean(generated_validity) - torch.mean(real_validity)


def compute_spectral_loss(
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
) -> torch.Tensor:
    """Compute L1 loss between spectrograms."""
    return torch.mean(torch.abs(real_specs - generated_specs))


def compute_spectral_convergence_loss(
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
) -> torch.Tensor:
    """Compute spectral convergence loss between spectrograms."""
    numerator = torch.norm(generated_specs - real_specs, p=2)
    denominator = torch.norm(real_specs, p=2) + 1e-8
    return numerator / denominator


def compute_gradient_penalty(
    critic: Critic,
    generated_specs: torch.Tensor,
    real_specs: torch.Tensor,
) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP."""
    alpha = torch.rand(model_params.BATCH_SIZE, 1, 1, 1).to(model_params.DEVICE)
    interpolates = (real_specs * alpha + generated_specs * (1 - alpha)).requires_grad_(
        True
    )
    c_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(c_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(model_params.BATCH_SIZE, -1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1) ** 2).mean()


def train_epoch(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_C: torch.optim.Optimizer,
    epoch_number: int,
) -> dict[str, float]:
    """Train one GAN epoch."""
    # Prepare loop
    generator.train()
    critic.train()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0

    # Training loop
    for i, (real_spec,) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Train Epoch {epoch_number+1}",
    ):
        for _ in range(model_params.CRITIC_STEPS):
            real_spec = real_spec.to(model_params.DEVICE)
            optimizer_C.zero_grad()
            z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM).to(
                model_params.DEVICE
            )

            generated_spec = generator(z)
            real_validity = critic(real_spec)
            fake_validity = critic(generated_spec.detach())

            c_loss = compute_critic_loss(
                critic,
                fake_validity,
                real_validity,
                generated_spec,
                real_spec,
                True,
            )

            c_loss.backward()
            optimizer_C.step()

            total_c_loss += c_loss.item()
            w_dist = compute_wasserstein_loss(real_validity, fake_validity)
            total_w_dist += w_dist.item()

        # Train generator
        if i % model_params.CRITIC_STEPS == 0:
            optimizer_G.zero_grad()
            z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM).to(
                model_params.DEVICE
            )

            generated_spec = generator(z)
            generated_validity = critic(generated_spec)
            g_loss = compute_generator_loss(
                generated_validity, generated_spec, real_spec
            )

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer_G.step()

            total_g_loss += g_loss.item()

    avg_g_loss = total_g_loss / (len(dataloader) // model_params.CRITIC_STEPS)
    avg_c_loss = total_c_loss / (len(dataloader) * model_params.CRITIC_STEPS)
    avg_w_dist = total_w_dist / (len(dataloader) * model_params.CRITIC_STEPS)

    return {
        "epoch": epoch_number,
        "g_loss": avg_g_loss,
        "c_loss": avg_c_loss,
        "w_dist": avg_w_dist,
    }


def validate(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    epoch_number: int,
) -> dict[str, float | torch.Tensor]:
    """Execute validation pass."""
    # Prepare loop
    generator.eval()
    critic.eval()
    total_g_loss, total_c_loss, total_w_dist = 0.0, 0.0, 0.0
    total_fad, total_is, total_kid = 0.0, 0.0, 0.0

    # Validation loop
    with torch.no_grad():
        for _, (real_spec,) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Val Epoch {epoch_number+1}",
            disable=True,
        ):
            real_spec = real_spec.to(model_params.DEVICE)
            z = torch.randn(model_params.BATCH_SIZE, model_params.LATENT_DIM).to(
                model_params.DEVICE
            )
            generated_spec = generator(z)

            real_validity = critic(real_spec)
            generated_validity = critic(generated_spec)

            g_loss = compute_generator_loss(
                generated_validity, generated_spec, real_spec
            )
            c_loss = compute_critic_loss(
                critic,
                generated_validity,
                real_validity,
                generated_spec,
                real_spec,
                False,
            )

            metrics = calculate_audio_metrics(real_spec, generated_spec)

            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()
            total_w_dist += compute_wasserstein_loss(
                real_validity, generated_validity
            ).item()
            total_fad += metrics["fad"]
            total_is += metrics["is"]
            total_kid += metrics["kid"]

    return {
        "epoch": epoch_number,
        "g_loss": total_g_loss / len(dataloader),
        "c_loss": total_c_loss / len(dataloader),
        "w_dist": total_w_dist / len(dataloader),
        "fad": total_fad / len(dataloader),
        "is": total_is / len(dataloader),
        "kid": total_kid / len(dataloader),
        "val_specs": generated_spec.cpu(),
    }


def training_loop(train_loader: DataLoader, val_loader: DataLoader) -> None:
    """Execute complete training loop."""
    print("Starting training for", model_params.selected_model)
    # Initialize models
    generator = Generator(stage=0)
    critic = Critic(stage=0)

    optimizer_G = torch.optim.AdamW(
        generator.parameters(),
        lr=model_params.LR_G,
        weight_decay=0.02,
        betas=(0.0, 0.99),
    )
    optimizer_C = torch.optim.AdamW(
        critic.parameters(), lr=model_params.LR_C, weight_decay=0.02, betas=(0.0, 0.99)
    )
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode="min", factor=0.1, patience=3)
    scheduler_C = ReduceLROnPlateau(optimizer_C, mode="min", factor=0.1, patience=3)

    generator.to(model_params.DEVICE)
    critic.to(model_params.DEVICE)

    best_metrics = {
        "fad": float("inf"),
        "is": float("-inf"),
        "kid": float("inf"),
    }
    epochs_no_improve = 0

    for epoch in range(model_params.N_EPOCHS):
        # Train
        train_metrics = train_epoch(
            generator,
            critic,
            train_loader,
            optimizer_G,
            optimizer_C,
            epoch,
        )

        # Validate
        val_metrics = validate(generator, critic, val_loader, epoch)

        # Step scheduler and models
        scheduler_G.step(val_metrics["fad"])
        scheduler_C.step(val_metrics["fad"])
        generator.progress_step()
        critic.progress_step()

        # Print information
        print(
            f"TRAIN g_loss: {train_metrics['g_loss']:.4f} c_loss: {train_metrics['c_loss']:.4f} w_dist: {train_metrics['w_dist']:.4f}"
        )
        print(
            f"VAL g_loss: {val_metrics['g_loss']:.4f} c_loss: {val_metrics['c_loss']:.4f} w_dist: {val_metrics['w_dist']:.4f}"
        )
        print(
            f"VAL FAD: {val_metrics['fad']:.4f} IS: {val_metrics['is']:.4f} KID: {val_metrics['kid']:.4f}"
        )

        # End of epoch handling
        if val_metrics["fad"] < best_metrics["fad"]:
            best_metrics["fad"] = val_metrics["fad"]
            best_metrics["is"] = val_metrics["is"]
            best_metrics["kid"] = val_metrics["kid"]
            epochs_no_improve = 0
            DataUtils.visualize_spectrogram_grid(
                val_metrics["val_specs"],
                f"Raw Model Output Epoch {epoch+1} - w_dist: {val_metrics['w_dist']:.4f} FAD: {val_metrics['fad']:.4f} IS: {val_metrics['is']:.4f} KID: {val_metrics['kid']:.4f}",
                f"static/{model_selection.name.lower()}_best_val_spectrograms.png",
            )
            model_utils.save_model(generator)
            print(
                f"New best model saved with metrics - FAD: {val_metrics['fad']:.4f} IS: {val_metrics['is']:.4f} KID: {val_metrics['kid']:.4f} -- Stage {generator.stage}/{ModelParams.MAX_STAGE}"
            )
        else:
            epochs_no_improve += 1
            print(f"Epochs without improvement: {epochs_no_improve}")

        if epochs_no_improve >= model_params.PATIENCE:
            print("Early stopping triggered")
            break
