from utils.helpers import ModelParams, ModelUtils

params = ModelParams()
utils = ModelUtils(params.sample_length)

count_to_generate = 2

utils.generate_audio(params.model_save_path, count_to_generate)
