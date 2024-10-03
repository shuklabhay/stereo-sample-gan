import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.file_helpers import (
    load_loudness_data,
)
from utils.signal_helpers import encode_sample_directory
from usage_params import UsageParams

# Encode audio samples
params = UsageParams()
if len(sys.argv) > 1:
    visualize = sys.argv[1].lower() == "visualize"
else:
    visualize = False


encode_sample_directory(params.training_audio_dir, params.compiled_data_path, visualize)

real_data = load_loudness_data(
    params.compiled_data_path
)  # datapts, channels, frames, freq bins
print(f"{params.training_audio_dir} data shape: {str(real_data.shape)}")
