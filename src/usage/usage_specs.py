# Processing training data
training_audio_dir = "data/one_shots"  # Your training data path
compiled_data_path = "data/compiled_data.npy"  # Your compiled data/output path
training_sample_length = 0.6  # seconds

# Saving model
outputs_dir = "outputs"  # Where to save your generated audio & model
model_save_name = "StereoSampleGAN-OldKick"  # What to name your model save
model_save_path = f"{outputs_dir}/{model_save_name}.pth"

# Generating audio
model_to_generate_with = model_save_path  # Generation model path
audio_generation_count = 2  # Audio examples to generate
generated_audio_name = "generated_audio"  # Output file name
generated_sample_length = 0.6  # Match model training data audio length
visualize_generated = True  # SHow generated audio spectrogra,s
