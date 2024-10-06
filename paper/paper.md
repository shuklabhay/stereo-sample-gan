# StereoSampleGAN: A Computationally Inexpensive Approach High Fidelity Stereo Audio Generation.

Abhay Shukla\
abhayshuklavtr@gmail.com\
Continuation of UCLA COSMOS 2024 Research

## 1. Abstract

Existing convolutional aproaches to audio generation often are limited to producing low-fidelity, single-channel, monophonic audio, while demanding significant computational resources for both training and inference. To address these challenges, this work introduces StereoSampleGAN, a novel audio generation architecture that combines a Deep Convolutional Wasserstein GAN (WGAN), attention mechanisms, and loss optimization techniques. StereoSampleGAN allows high-fidelity, stereo audio generation for audio samples while being remaining computationally efficient. Training on three distinct sample datasets with varying spectral overlap–two of kick drums and one of tonal one shots–StereoSampleGAN demonstrates promising results in generating high quality simple stereo sounds. While successfully understanding how to generate the "shape" of required audio, it displays notable limiatations in achieving the correct "tone," in some cases even generating incoherent noise. These results indicate finite limitations and areas for improvement to this approach of audio generation.

## 2. Introduction

## 3. Data Manipulation

## 3.1 Datasets

This paper utilizes three distinct data sets engineered to measure the model's resilince to variation in spectral content.

1. Curated Kick Drum Set: Kick drum impulses with primarily short decay profiles.

2. Diverse Kick Drum Set: Kick drum impulses with greater variation in decay profile and overall harmonic content.

3. Instrument One Shot Set: Single note impulses capturing the tonal qualities and spectral characteristics of varying synthesizer and instrument sounds.

These datasets provide robust frameworks for determining the model's response to varying amounts of variation within training data. Most audio is sourced from online "digital audio production sample packs" which compile sounds for a wide variety of generes and use cases.

## 3.2 Feature Extraction and Encoding

## 4. Model Implementation

### 4.1. Architecture

### 4.2. Training

## 5. Results and Discussion

### 5.1. Evaluation

The model generated 44.1k high quality audio, but not audio of high quality (important distinction). Shape vs tone (fundamental completely missing), why it makes sense (limitations to ft, training for shape of img not AUDIO)

### 5.2. Contributions

## 6. Conclusion

## 7. References
