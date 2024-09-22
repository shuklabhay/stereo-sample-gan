import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.file_helpers import (
    load_loudness_data,
)
from utils.signal_helpers import encode_sample_directory

# Encode audio samples
if len(sys.argv) > 1:
    visualize = sys.argv[1].lower() == "visualize"
else:
    visualize = False


training_audio_dir = ""  # Your training data path
compiled_data_path = "data/compiled_data.npy"  # Your compiled data/output path
audio_sample_length = 0.6  # 600 ms

encode_sample_directory(training_audio_dir, compiled_data_path, visualize)

real_data = load_loudness_data(
    compiled_data_path
)  # datapts, channels, frames, freq bins
print("Data shape:", str(real_data.shape))
