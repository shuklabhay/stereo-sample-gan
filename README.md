# StereoSampleGAN

[![On Push](https://github.com/shuklabhay/stereo-sample-gan/actions/workflows/push.yml/badge.svg)](https://github.com/shuklabhay/stereo-sample-gan/actions/workflows/push.yml/badge.svg)

StereoSampleGAN: A lightweight approach high fidelity stereo audio sample generation.

## Model Usage

1. Prereqs

- Optional but highly reccomended: Set up a [Python virtual environment.](https://www.youtube.com/watch?v=e5GL1obY_sI)
  - Audio loader package `librosa` requires an outdated version of Numpy
- Install requirements by running `pip3 install -r requirements.txt`

2. Generate Audio

- Specify usage paramaters in `usage_params.py`
  - Make sure `generated_sample_length` value matches what the model was trained on
  - See pretrained models section for more info
- Generate audio by running `python3 generate.py`

3. Train model

- Specify training data paramaters in `usage_params.py`
  - I reccomend anywhere between 4,000-8000 training examples, any multiple of 8
- Process training data by running `python3 encode_audio_data.py`
- Train model by running `python3 stereo_sample_gan.py`

## Pretrained Models

### Diverse Kick Drums

Kick Drum generation model trained on ~8000 essentially random kick drums.

- More variation between each generated sample but audio is often inconsistent and contains some artifacts.
- `model_save_name="StereoSampleGAN-DiverseKick`
- `training_sample_length = 0.6`

Training progress:

<img src="paper/static/diverse_kick_training_progress.gif" alt="Diverse kick training progress" width="400">

### Diverse Kick Drums

Kick Drum generation model trained on ~4000 curated kick drums.

- Less variation between each drum sample but also less noisy and closer to the "normal" kick drum sound
- `model_save_name="StereoSampleGAN-CuratedKick`
- `training_sample_length = 0.6`

### One Shots

- Instrument one shot generation model, trained on ~3000 semi-curated instrument one shots.
- WIP
- `model_save_name="StereoSampleGAN-InstrumentOneShot`
- WIP

## Directories

- `outputs`: Trained model and generated audio
- `paper`: Research paper / model writeup
- `static`: Static images and gifs
- `src`: Model source code
  - `utils`: Model and data utilities
  - `data_processing`: Training data processing scripts
