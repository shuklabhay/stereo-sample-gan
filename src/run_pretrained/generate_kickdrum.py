import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import ModelType
from utils.helpers import ModelParams, ModelUtils

params = ModelParams()
utils = ModelUtils(params.sample_length)

params.load_params(ModelType.KICKDRUM)
count_to_generate = 2

utils.generate_audio(params.model_save_path, count_to_generate)
