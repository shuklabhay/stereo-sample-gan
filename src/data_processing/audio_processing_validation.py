import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import DataUtils, ModelParams, SignalProcessing

model_params = ModelParams()
signal_processing = SignalProcessing(model_params.sample_length)


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute mean squared error between two waveforms."""
    return F.mse_loss(reconstructed, original).item()


def main():
    parser = argparse.ArgumentParser(description="Validate STFT reconstruction MSE")
    parser.add_argument(
        "--eval_path",
        "-p",
        type=str,
        default=model_params.training_audio_dir,
        help="Audio file or directory to process",
    )
    args = parser.parse_args()

    files = DataUtils.list_audio_files(args.eval_path)
    if not files:
        print(f"No audio files found at {args.eval_path}")
        return

    total_mse = 0.0
    for i, audio_path in enumerate(tqdm(files, desc="Processing examples")):
        # Find original waveform
        waveform_np = DataUtils.load_audio(audio_path, sample_length=0.6)
        waveform = torch.from_numpy(waveform_np)
        print(waveform_np.min(), waveform_np.max())
        # Find reconstructed waveform
        result = signal_processing.stft_and_istft(audio_path, f"eval_{i}")
        reconstructed = torch.from_numpy(result["istft"])

        # Calculate MSE
        mse = compute_mse(waveform, reconstructed)
        total_mse += mse

    avg_mse = total_mse / len(files)
    print(f"Processed {len(files)} files, Average Reconstruction MSE = {avg_mse:.6f}")


if __name__ == "__main__":
    main()
