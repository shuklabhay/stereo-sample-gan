import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from utils.constants import ModelParams, SignalConstants
from utils.helpers import DataUtils, ModelParams

matplotlib.use("agg")


def calculate_audio_metrics(
    real_specs: torch.Tensor, generated_specs: torch.Tensor
) -> dict:
    """Calculate FAD, ISCORE, KID, and Phase Coherence."""
    model_params = ModelParams()
    fad_value = calculate_fad(model_params, real_specs, generated_specs)
    inception_dist = calculate_inception_score(model_params, generated_specs)
    kid_value = calculate_kernel_inception_distance(
        model_params, real_specs, generated_specs
    )
    swi_real = calculate_swi(real_specs)
    swi_gen = calculate_swi(generated_specs)

    return {
        "fad": fad_value,
        "kid": kid_value,
        "is": inception_dist,
        "swi": swi_gen,
        "swi_real": swi_real,
    }


def calculate_fad(
    model_params: ModelParams, real_specs: torch.Tensor, generated_specs: torch.Tensor
) -> float:
    """Calculate Fréchet Audio Distance."""
    vggish = VGGISH.get_model().to(model_params.DEVICE)

    real_specs = torch.tensor(
        DataUtils.scale_data_to_range(real_specs.detach().cpu().numpy(), -1, 1),
        device=model_params.DEVICE,
    )
    real_specs = F.interpolate(
        real_specs.mean(dim=1, keepdim=True),
        size=(96, 64),
        mode="bicubic",
    )
    generated_specs = torch.tensor(
        DataUtils.scale_data_to_range(generated_specs.detach().cpu().numpy(), -1, 1),
        device=model_params.DEVICE,
    )
    generated_specs = F.interpolate(
        generated_specs.mean(dim=1, keepdim=True),
        size=(96, 64),
        mode="bicubic",
    )

    with torch.no_grad():
        real_feats = vggish(real_specs)
        generated_feats = vggish(generated_specs)

    mu_real = real_feats.mean(0)
    sigma_real = torch.cov(real_feats.T)
    mu_generated = generated_feats.mean(0)
    sigma_generated = torch.cov(generated_feats.T)
    fad_val = torchaudio.functional.frechet_distance(
        mu_real, sigma_real, mu_generated, sigma_generated
    )

    return fad_val.item()


def calculate_inception_score(
    model_params: ModelParams, generated_specs: torch.Tensor
) -> float:
    """Calculate Inception Score for generated audio spectrograms."""
    # Process generated specs
    generated_mono = generated_specs.mean(dim=1, keepdim=True)
    generated_scaled = (generated_mono + 1) / 2
    generated_scaled = F.interpolate(generated_scaled, size=(299, 299), mode="bicubic")
    generated_rgb = generated_scaled.repeat(1, 3, 1, 1)

    # Calculate Inception Score
    inception_score = InceptionScore(splits=10, normalize=True).to(model_params.DEVICE)
    inception_score.update(generated_rgb)
    score = inception_score.compute()
    return score[0].item()


def calculate_kernel_inception_distance(
    model_params: ModelParams, real_specs: torch.Tensor, generated_specs: torch.Tensor
) -> float:
    """Calculate KID between real and generated spectrograms."""
    # Process generated specs
    generated_mono = generated_specs.mean(dim=1, keepdim=True)
    generated_scaled = (generated_mono + 1) / 2
    generated_scaled = F.interpolate(generated_scaled, size=(299, 299), mode="bicubic")
    generated_rgb = generated_scaled.repeat(1, 3, 1, 1)

    # Process real specs
    real_mono = real_specs.mean(dim=1, keepdim=True)
    real_scaled = (real_mono + 1) / 2
    real_scaled = F.interpolate(real_scaled, size=(299, 299), mode="bicubic")
    real_rgb = real_scaled.repeat(1, 3, 1, 1)

    # Calculate KID
    subset_size = min(real_specs.shape[0], generated_specs.shape[0], 50)
    kid = KernelInceptionDistance(
        subsets=100, subset_size=subset_size, normalize=True
    ).to(model_params.DEVICE)
    kid.update(real_rgb, real=True)
    kid.update(generated_rgb, real=False)
    kid_mean, _ = kid.compute()
    return kid_mean.item()


def calculate_swi(specs: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute Stereo Width Index for a batch of 2‑ch spectrograms."""
    left, right = specs[:, 0], specs[:, 1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    energy_mid = torch.mean(mid**2)
    energy_side = torch.mean(side**2)
    return (energy_side / (energy_mid + eps)).item()


def visualize_pair_spectrogram_grid(
    real_specs: torch.Tensor,
    gen_specs: torch.Tensor,
    title: str,
    save_path: str,
    items: int = 8,
) -> None:
    """Plot paired spectrogram and pan information."""
    real_arr = real_specs.detach().cpu().numpy()
    gen_arr = gen_specs.detach().cpu().numpy()
    rows = min(items, max(len(real_arr), len(gen_arr)))
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2))
    fig.suptitle(title)
    fig.subplots_adjust(top=0.85, hspace=0.4)

    freq_bins = np.arange(real_arr.shape[-1])
    for i in range(rows):
        # Show lr specs
        if i < len(real_arr):
            rl, rr = real_arr[i]
            axes[i, 0].imshow(rl.T, origin="lower", aspect="auto", cmap="viridis")
            axes[i, 1].imshow(rr.T, origin="lower", aspect="auto", cmap="viridis")
        else:
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")

        # Show width scatter plot
        if i < len(real_arr):
            L2 = real_arr[i][0] ** 2
            R2 = real_arr[i][1] ** 2
            pan = L2.sum(axis=0) / (L2 + R2 + 1e-8).sum(axis=0)
            axes[i, 2].scatter(pan, freq_bins, color="C0", s=6)
            axes[i, 2].set_xlim(0, 1)
            axes[i, 2].set_ylim(0, freq_bins[-1])
            axes[i, 2].set_xlabel("Pan")
            axes[i, 2].set_ylabel("Freq bin")
            axes[i, 2].set_title("Real Pan", fontsize=8)
        else:
            axes[i, 2].axis("off")

        # Show l/r specs
        if i < len(gen_arr):
            gl, gr = gen_arr[i]
            axes[i, 3].imshow(gl.T, origin="lower", aspect="auto", cmap="viridis")
            axes[i, 4].imshow(gr.T, origin="lower", aspect="auto", cmap="viridis")
        else:
            axes[i, 3].axis("off")
            axes[i, 4].axis("off")

        # Show width scatter plot
        if i < len(gen_arr):
            L2 = gen_arr[i][0] ** 2
            R2 = gen_arr[i][1] ** 2
            pan = L2.sum(axis=0) / (L2 + R2 + 1e-8).sum(axis=0)
            axes[i, 5].scatter(pan, freq_bins, color="C1", s=6)
            axes[i, 5].set_xlim(0, 1)
            axes[i, 5].set_ylim(0, freq_bins[-1])
            axes[i, 5].set_xlabel("Pan")
            axes[i, 5].set_ylabel("Freq bin")
            axes[i, 5].set_title("Gen Pan", fontsize=8)
        else:
            axes[i, 5].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
