# Kick it Out: Shortcomings to Generating Audio Representations With a Deep Convolution Generative Network

Abhay Shukla\
abhayshuklavtr@gmail.com\
Continuation of UCLA COSMOS 2024 Research

## Abstract

## Introduction

Since their introduction, CNN based Generative Adversarial Networks (DCGANs) have vastly increased the capabilites of machine learning models, allowing high-fidelity synthetic image generation [1], but requiring aditional optimizations for audio generation [2]. To generate high quality audio, models must capture temporal relationships and spectral characteristcs, and be able to replicate it without inconsistencies that could go unnoticed in an image and be apparent in audio. Accounting for these complexities requires additional modifications and straying away from the pure DCGAN architecture. This work attempts to recognize the limitations of audio representation generation using **only** Deep Convolution in a Generative Network.

Kick drums are used here as the sound to generate because they best fit the criteria of complexity, tonality, length, and temporal patterns. They are simple sounds that have the potential to have lots of variance. Kick drums are also an integral part of digital audio production and the foundational element of almost every song and drumset. Due to their importance, finding a large quantity of high quality, unique kick drum samples is often a problem in the digital audio production enviroment.

This investigation primarily seeks to determine how feasible it can be to use purely a DCGAN Architecture to recognize and replicate the spatial patterns and temporal patterns of an image representation of a kick drum. We will also experiment with generating pure sine waves as a means of validation.

## Methodology

### Data Collection

Training data is first sourced from digital production “sample packs” compiled by various parties. These packs contain a variety of kick drum samples (analog, cinematic, beatbox, heavy, edm, etc), providing a wholstic selection of samples that for the most part include a set of "defining characteristics" of a kick drum.

The goal of this model is to replicate the following characteristics of a kick drum: [graphic kick drum spectrogram]

- A specific length audio sample
- An atonal transient “click” at the beginning of the generated audio incorporating most of the frequency spectrum
- A sustained, decaying low "rumble" following the transient of the sample
- An overall "decaying" nature

### Feature Extraction/Encoding

The training data used is a compilation of 7856 audio samples split into batches of 8. Each sample is normalized to a length of 500 miliseconds and passed into a Short-time Fourier Transform, returning a representation of audio as an array of amplitudes for 2 channels, 176 frames of audio, 257 frequency bins.

While amplitude data is important, this data is by nature skewed towards lower frequencies which contain more intensity. To account for this, a few things are done. First, after extracting channel amplitudes, the whole tensor of data is scaled to be between 0 and 100. The data is then passed through a noise threshold where all values under 10e-10 are set to zero and this normalized, noise gated amplitude information is converted into a logarithmic, decibal scale. The decibal scale describes percieved loudness instead of intensity, displaying audio information in a more uniform way relative to the entire frequency spectrum. This data is then finally scaled to be between -1 and 1, representative of the output the model creates using the hyperbolic tangent activation function.

[show amp data vs loudness data graph]

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
