from file_helpers import (
    audio_data_dir,
    compiled_data_path,
    load_npy_data,
)
from signal_helpers import encode_sample_directory

# Encode samples
encode_sample_directory(audio_data_dir, silent=False)

real_data = load_npy_data(compiled_data_path)  # datapts, channels, frames, freq bins
print("Data " + str(real_data.shape))
