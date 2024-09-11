import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.file_helpers import (
    audio_data_dir,
    compiled_data_path,
    load_loudness_data,
)
from utils.signal_helpers import encode_sample_directory

# Encode samples
if len(sys.argv) > 1:
    visualize = sys.argv[1].lower() == "visualize"
else:
    visualize = False

encode_sample_directory(audio_data_dir, visualize)

real_data = load_loudness_data(
    compiled_data_path
)  # datapts, channels, frames, freq bins
print("Data shape,", str(real_data.shape))
