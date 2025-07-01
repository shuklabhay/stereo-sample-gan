import argparse

from torch.utils.data import DataLoader, TensorDataset, random_split

from train import training_loop
from utils.constants import ModelParams, ModelType
from utils.helpers import ModelUtils, SignalProcessing


def main():
    parser = argparse.ArgumentParser(description="Train GAN on kicks or snares")
    parser.add_argument(
        "--model_type",
        choices=[t.value for t in ModelType],
        required=True,
        help="Select model type: Kickdrum or Snaredrum",
    )
    args = parser.parse_args()

    model_type = ModelType(args.model_type)
    model_params = ModelParams()
    model_params.load_params(model_type)
    model_utils = ModelUtils(model_params.sample_length)
    signal_processing = SignalProcessing(model_params.sample_length)

    # Encode data from directory
    all_spectrograms = signal_processing.encode_sample_directory(
        model_params.training_audio_dir, model_params.selected_model
    )

    # Create train and val datasets
    train_size = int(0.9 * len(all_spectrograms))
    val_size = len(all_spectrograms) - train_size
    train_dataset, val_dataset = random_split(
        TensorDataset(all_spectrograms), [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        num_workers=16,
        batch_size=model_params.BATCH_SIZE,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=16,
        batch_size=model_params.BATCH_SIZE,
        drop_last=True,
    )

    # Train model
    training_loop(train_loader, val_loader)


if __name__ == "__main__":
    main()
