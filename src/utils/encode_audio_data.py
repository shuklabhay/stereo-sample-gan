from utils.file_helpers import (
    audio_data_dir,
    average_spectrogram_path,
    compiled_data_path,
    load_npy_data,
)
from utils.signal_helpers import encode_sample_directory

# Encode samples
encode_sample_directory(audio_data_dir, silent=True)

real_data = load_npy_data(compiled_data_path)  # datapts, channels, frames, freq bins
print("Data " + str(real_data.shape))
