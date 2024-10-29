from utils.helpers import ModelParams, ModelUtils


params = ModelParams()
utils = ModelUtils(params.sample_length)

params.load_params("Kickdrum")
count_to_generate = 2

utils.generate_audio(params.model_save_path, count_to_generate, True)
