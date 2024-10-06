# StereoSampleGAN

[![On Push](https://github.com/shuklabhay/stereo-sample-gan/actions/workflows/push.yml/badge.svg)](https://github.com/shuklabhay/stereo-sample-gan/actions/workflows/push.yml/badge.svg)

StereoSampleGAN: A computationally inexpensive approach high fidelity stereo audio sample generation.

## Model Usage

### 1. Prereqs

- Optional but highly reccomended: Set up a [Python virtual environment.](https://docs.python.org/3/library/venv.html)
  - Audio loader package `librosa` requires an outdated version of Numpy
- Install requirements by running `pip3 install -r requirements.txt`

### 2. Generate audio from pretrained models

Specify sample count to generate, output, etc `usage_params.py`

- Generate audio from the Curated Kick model by running `python3 src/run_pretrained/generate_curated_kick.py`
- Generate audio from the Diverse Kick model by running `python3 src/run_pretrained/generate_diverse_kick.py`
- Generate audio from the Instrument One Shot model by running `python3 src/run_pretrained/generate_instrument_one_shot.py`

### 3. Train model

Specify training data paramaters in `usage_params.py`

- I reccomend anywhere between 4,000-8000 training examples, any multiple of 8 and audio
  <1.5 sec long (longer hasn't been fully tested)
- Prepare training data by running `python3 src/data_processing/encode_audio_data.py`
- Train model by running `python3 src/stereo_sample_gan.py`
- Generate audio (based on current `usage_params.py`) by running `python3 src/generate.py`

Training progress visualization (training Diverse Kick Drum Model):

<img src="static/diverse_kick_training_progress.gif" alt="Diverse kick training progress" width="400">

## Pretrained Models

### Diverse Kick Drum

Kick drum generation model trained on ~8000 essentially random kick drums.

- More variation between each generated sample, audio is occasionally inconsistent and noisy.

<img src="static/diverse_kick_generated_examples.png" alt="Diverse kick model generated examples" width="800">

### Curated Kick Drum

Kick drum generation model trained on ~4400 kick drums with closer matching overall characteristics.

- Less variation between each drum sample's decay and auditory tone.

<img src="static/curated_kick_generated_examples.png" alt="Curated kick model generated examples" width="800">

### Instrument One Shot

Instrument one shot generation model, trained on ~3000 semi-curated instrument one shots.

- Demonstrates model's capability to generate longer audio, yet fails to generate coherent and useable instrument one shots.

<img src="static/instrument_one_shot_generated_examples.png" alt="Instrument one shot model generated examples" width="800">

## Directories

- `outputs`: Trained model and generated audio
- `paper`: Research paper / model writeup
- `static`: Static resources
- `src`: Model source code
  - `utils`: Model and data utilities
  - `data_processing`: Training data processing scripts
