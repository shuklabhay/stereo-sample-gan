# Kick it Out: Shortcomings to Generating Audio Representations With a Deep Convolution Generative Network

Abhay Shukla\
abhayshuklavtr@gmail.com\
Continuation of UCLA COSMOS 2024 Research

## Abstract

## Introduction

Since their introduction, CNN based Generative Adversarial Networks (DCGANs) have vastly increased the capabilites of machine learning models, allowing high-fidelity synthetic image generation [1], but requiring aditional optimizations for audio generation [2]. To generate high quality audio, models must capture temporal relationships and spectral characteristcs, and be able to replicate it without inconsistencies that could go unnoticed in an image and be apparent in audio. Accounting for these complexities requires additional modifications and straying away from the pure DCGAN architecture. This work attempts to recognize the limitations of audio representation generation using **only** Deep Convolution in a Generative Network.

This project uses kick drums as the sound to generate since they best fit the criteria of complexity, tonality, length, and temporal patterns. Kick drums are also an integral part of digital audio production and the foundational element of almost every song and drumset. Due to their importance, finding a large quantity of high quality, unique kick drum samples is often a problem in the digital audio production enviroment.

This investigation specifically seeks to determine how feasible it can be to use a DCGAN Architecture to recognize and replicate the spatial patterns and temporal patterns of an image representation of a kick drum. We will also experiment with pure sine wave validation at one frequency.

## Methodology

### Data Collection and Processing

Training data is first sourced from digital production “sample packs” compiled by various parties. These packs contain various amounts of analog, cinematic, heavy, and edm kick drum samples, providing a wholstic yet random selection of kick drums that loosely contain the same characteristics.

The goal of this model is to replicate the following characteristics of a kick drum:

- A specific length audio sample
- An atonal transient “click” at the beginning of the generated audio incorporating most of the frequency spectrum
- A sustained, decaying low "rumble" following the transient of the sample
- The overall "decaying" nature of a kick drum

This work uses 7856 data points split into batches of eight. Each audio sample is normalized to a length of 500 miliseconds and then

old:
Every audio clip is normalized to 500 ms and passed into a Short-time Fourier Transform, returning amplitudes for frequency bins at every frame of audio for each channel. To amplify audio features, the amplitude data is then passed through a noise floor, decibel scaled, and rescaled to be between -1 and 1.

### Model Architecture

## Results

### Kick Drum Generation

### Sine Validation

## Discussion

## Conclusion

## References

<a id="1">[1]</a> CNN based GAN
https://arxiv.org/abs/1511.06434

<a id="2">[2]</a> GAN audio generation
https://arxiv.org/abs/1802.04208

similar result to me
https://openaccess.thecvf.com/content_CVPR_2020/papers/Durall_Watch_Your_Up-Convolution_CNN_Based_Generative_Deep_Neural_Networks_Are_CVPR_2020_paper.pdf

STRUCTURE OF A PAPER (claude generated)

1. title: done
2. Abstract: A brief summary of your paper, including the problem, methods, key results, and conclusions.
3. Introduction: Present the research problem, its importance, and your objectives.
4. Background/Literature Review: Provide context on deep convolution and its applications in audio generation. Review relevant previous work.
5. Methodology: Describe your approach, including:

- Neural network architecture
- Dataset description
- Training process
- Evaluation metrics

6. Results: Present your findings, including:

- Performance metrics
- Audio samples (if possible)
- Comparisons with other methods

7. Discussion: Interpret your results, discuss limitations, and suggest future work.
8. Conclusion: Summarize your key findings and their implications.
9. References: List all sources cited in your paper.
