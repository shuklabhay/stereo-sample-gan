import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import ModelParams, ModelUtils


params = ModelParams()
utils = ModelUtils(params.sample_length)

params.load_params("ChordShot")
count_to_generate = 2

utils.generate_audio(params.model_save_path, count_to_generate, True)
