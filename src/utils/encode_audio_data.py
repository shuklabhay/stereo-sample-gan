from helpers import (
    audio_data_dir,
    compiled_data_path,
    encode_sample_directory,
    load_npy_data,
)

# Encode samples
encode_sample_directory(audio_data_dir, silent=True)

real_data = load_npy_data(compiled_data_path)  # datapts, channels, frames, freq bins
print(real_data.shape)
