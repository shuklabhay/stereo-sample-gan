import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.file_helpers import (
    audio_data_dir,
    compiled_data_path,
    load_loudness_information,
)
from utils.signal_helpers import encode_sample_directory

# Encode samples
if len(sys.argv) > 1:
    visualize = sys.argv[1].lower() == "true"
else:
    visualize = False

encode_sample_directory(audio_data_dir, visualize)

real_data = load_loudness_information(
    compiled_data_path
)  # datapts, channels, frames, freq bins
print("Data " + str(real_data.shape))
