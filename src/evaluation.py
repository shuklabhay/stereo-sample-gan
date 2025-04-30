import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import ModelParams, ModelType
from utils.eval_helpers import calculate_audio_metrics, visualize_pair_spectrogram_grid
from utils.helpers import DataUtils, ModelUtils, SignalProcessing


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics on kick and snare samples"
    )
    parser.add_argument(
        "--eval_kicks_dir", type=str, required=True, help="Dir of real kickdrums"
    )
    parser.add_argument(
        "--eval_snares_dir", type=str, required=True, help="Dir of real snaredrums"
    )
    parser.add_argument(
        "--count", type=int, default=500, help="Number of samples per type"
    )
    args = parser.parse_args()

    # Prepare for evaluation
    params = ModelParams()
    signal_proc = SignalProcessing(params.sample_length)
    utils = ModelUtils(params.sample_length)

    # Evaluate both types
    for label, real_dir, model_type in [
        ("Kickdrum", args.eval_kicks_dir, ModelType.KICKDRUM),
        ("Snaredrum", args.eval_snares_dir, ModelType.SNAREDRUM),
    ]:
        # Generate spectrograms
        params.load_params(model_type)
        generated_result = utils.generate_audio(
            params.model_save_path, args.count, save=False
        )
        gen_batch = torch.tensor(generated_result["specs"])[: args.count]

        # Load spectrograms
        files = DataUtils.list_audio_files(real_dir)[: args.count]
        real_specs = []
        for p in files:
            wav = DataUtils.load_audio(p, params.sample_length)
            real_specs.append(signal_proc.audio_to_norm_spec(wav))
        real_batch = torch.tensor(np.stack(real_specs))

        # Calculate and display metrics
        metrics = calculate_audio_metrics(real_batch, gen_batch)
        print(f"\n== {label.upper()} SCORES ==")
        print(f"FAD:      {metrics['fad']:.4f}")
        print(f"KID:      {metrics['kid']:.4f}")
        print(f"IS:       {metrics['is']:.4f}")
        print(f"SWI:      {metrics['swi']:.4f}")
        print(f"SWI_REAL: {metrics['swi_real']:.4f}")

        # per‐type real vs generated comparison
        per_img = os.path.join(params.outputs_dir, f"{label.lower()}_comparison.png")
        visualize_pair_spectrogram_grid(
            real_specs=real_batch,
            gen_specs=gen_batch,
            title=f"{label} Real vs Generated",
            save_path=per_img,
        )


if __name__ == "__main__":
    main()
