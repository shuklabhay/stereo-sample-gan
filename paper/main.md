# Kick it Out: Shortcomings to Generating Audio Representations With a Deep Convolution Generative Network

Abhay Shukla\
abhayshuklavtr@gmail.com\
Continuation of UCLA COSMOS 2024 Research

## Abstract

## Introduction

Kick drums are an integral part of digital audio production, the foundational element of almost every song and drumset. As a result, finding a large quantitity of high quality kick drum samples are often

The characteristics of the audio we're looking to replicate are as follows:

- An audio sample of 500 miliseconds long
- An with atonal transient “click” spanning a larger area of the frequency spectrum
- A sustained, decaying low "rumble" following the transient of the sample

Generative Adversarial Networks (GANs) have changed the landscape of the machine learning community, reaching new bounds in image generation [cite] and more recently natural language and audio generation [cite]. These audio generative models often employ Deep Convolutional GANs (DCGANs) to create spectrogram representations of audio.

This work aims specifically to generate kick drums using a similar DCGAN approach. [note that not using wgan or anything like that just using the dcgan how can i generate audio is it possible to generate kick drums]

This investigation specifically seeks to determine is a DCGAN Architecture can learn to recognize and replicate the spatial patterns and temporal patterns of an image representation kick drum.

## Related work

## Methodology

## Results

compare w/ wavegan?? ig

## Discussion

## Conclusion

## References

<a id="1">[1]</a> DCGAN paper methodology structure etc
https://arxiv.org/abs/1511.06434

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
