import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import DataUtils, ModelParams, SignalProcessing

model_params = ModelParams()
utils = DataUtils()
signal_processing = SignalProcessing(model_params.sample_length)


# Encode audio samples
if len(sys.argv) > 1:
    visualize = sys.argv[1].lower() == "visualize"
else:
    visualize = False


signal_processing.encode_sample_directory(
    model_params.training_audio_dir, model_params.compiled_data_path, visualize
)

real_data = utils.load_loudness_data(
    model_params.compiled_data_path
)  # datapts, channels, frames, freq bins
print(f"{model_params.training_audio_dir} data shape: {str(real_data.shape)}")
