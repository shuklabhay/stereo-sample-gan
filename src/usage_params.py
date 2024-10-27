# Main params
class UsageParams:
    def __init__(self):
        self.audio_generation_count = 2  # Audio examples to generate

        # Training params
        self.training_sample_length = 0.6  # seconds
        self.outputs_dir = "outputs"  # Where to save your generated audio & model

        self.model_save_name = (
            "StereoSampleGAN-CuratedKick"  # What to name your model save
        )
        self.training_audio_dir = "data/kick_samples_curated"  # Your training data path
        self.compiled_data_path = (
            "data/compiled_data.npy"  # Your compiled data/output path
        )
        self.model_save_path = f"{self.outputs_dir}/{self.model_save_name}.pth"

        # Generating audio
        self.model_to_generate_with = self.model_save_path  # Generation model path
        self.generated_audio_name = "generated_audio"  # Output file name
        self.visualize_generated = True  # Show generated audio spectrograms
