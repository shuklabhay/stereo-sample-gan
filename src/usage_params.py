# Main params
audio_generation_count = 2  # Audio examples to generate

# Training params
training_sample_length = 0.6  # seconds
outputs_dir = "outputs"  # Where to save your generated audio & model

model_save_name = "StereoSampleGAN-DiverseKick"  # What to name your model save
training_audio_dir = "data/kick_samples_diverse"  # Your training data path
compiled_data_path = "data/compiled_data.npy"  # Your compiled data/output path
model_save_path = f"{outputs_dir}/{model_save_name}.pth"

# Generating audio
model_to_generate_with = model_save_path  # Generation model path
generated_audio_name = "generated_audio"  # Output file name
visualize_generated = True  # Show generated audio spectrograms
