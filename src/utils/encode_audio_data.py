import sys
from file_helpers import (
    audio_data_dir,
    compiled_data_path,
    load_npy_data,
)
from signal_helpers import encode_sample_directory

# Encode samples
if len(sys.argv) > 1:
    visualize = sys.argv[1].lower() == "true"
else:
    visualize = False

encode_sample_directory(audio_data_dir, visualize)

real_data = load_npy_data(compiled_data_path)  # datapts, channels, frames, freq bins
print("Data " + str(real_data.shape))
