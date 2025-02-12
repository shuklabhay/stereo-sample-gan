import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from utils.constants import ModelParams
from utils.helpers import DataUtils, ModelParams, SignalProcessing


def calculate_audio_metrics(
    real_specs: torch.Tensor, generated_specs: torch.Tensor
) -> dict:
    """Calculate FAD, IS, KID, and Phase Coherence."""
    model_params = ModelParams()
    fad_value = calculate_fad(model_params, real_specs, generated_specs)
    inception_dist = calculate_inception_score(model_params, generated_specs)
    kid_value = calculate_kernel_inception_distance(
        model_params, real_specs, generated_specs
    )
    # phase_coherence = calculate_phase_coherence(generated_specs)

    return {
        "fad": fad_value,
        "is": inception_dist,
        "kid": kid_value,
        # "pc": phase_coherence,
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


def calculate_phase_coherence(generated_specs: torch.Tensor) -> float:
    """
    Computes left-right channel phase coherence for each generated sample,
    """
    sample_length = ModelParams().sample_length
    signal_processing = SignalProcessing(sample_length)
    phase_diffs = []
    for spec in generated_specs:
        # Convert to numpy audio
        audio = signal_processing.norm_spec_to_audio(spec.detach().numpy())
        left = audio[0]
        right = audio[1]

        # Extract phase
        left_phase = np.angle(scipy.signal.hilbert(left))
        right_phase = np.angle(scipy.signal.hilbert(right))

        # Calculate absolute phase difference
        diff = np.abs(left_phase - right_phase)
        diff_deg = np.degrees(diff)
        phase_diffs.append(diff_deg.cpu())

    avg_phase_diff_deg = np.mean(phase_diffs)
    return float(avg_phase_diff_deg)
