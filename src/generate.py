import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import ModelType
from utils.helpers import ModelParams, ModelUtils


def main():
    parser = argparse.ArgumentParser(description="Generate audio samples")
    parser.add_argument(
        "--type",
        choices=["kick", "snare"],
        default="kick",
        help="Type of audio to generate: kick or snare",
    )
    parser.add_argument(
        "--count", type=int, default=2, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
        help="Directory to save generated audio files",
    )
    args = parser.parse_args()

    params = ModelParams()
    utils = ModelUtils(params.sample_length)

    # Select the model type based on command line argument
    model_type = ModelType.KICKDRUM if args.type == "kick" else ModelType.SNARE
    params.load_params(model_type)

    # Make sure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    print(f"Generating {args.count} {args.type} drum samples...")
    utils.generate_audio(params.model_save_path, args.count, args.output_path)
    print(f"Generated {args.count} {args.type} samples in {args.output_path}/")


if __name__ == "__main__":
    main()
