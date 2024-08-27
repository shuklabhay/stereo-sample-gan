# StereoSampleGAN: Lightweight Stereo Audio Sample Generation

Abhay Shukla\
abhayshuklavtr@gmail.com\
Continuation of UCLA COSMOS 2024 Research

## basically yeah rewrite eveyrthing with the idea of this is happening for multiple types of audio and generative network for many kinds of audio

note that sample = nosies and drum instrumental things
note somewhere that like this is less for speech generation and more for like generating synths or samples or sounds. the novel thing is it uses 2 channels to generate w/ gan. like in theory it can learn to create one phrase again and again and again but not tested bc data constraints, sample gan. genrated audio samples.

go through and like rewrite everything make sure its all accurate and also like making sure whats important is all the things about the specific project like the new criteria of what this proejct is and whatever

## 1. Abstract

/ write this

Generate stereo instrumental audio (drums, snares/percs, instrument shots, etc orlike whatever i decide else to make), primarily use convolution/optmized to work on low resource situations

## 2. Introduction

Audio generation is an incredibly complex and computationally expensive task, and as architectures develop to efficiently process audio data, current audio generation models tend to reduce the sophistication of data, simplifying multi-channel signals into monophonic audio and reducing audio quality. These simplifications make audio data easier to process but trade off audio quality. While this work does not seek to generate audio indisginguisable from reality, it presents a unqiue approach to generating stero audio.

Audio generation models commonly take advantage of time-series optimized architectures (transformers, recurrent architectures, and HMMs [ADD CITATIONS FOR THIS- MAYBE JUST STH SAYING RECURRENT USED FOR AUDIO]), but this work instead opts to use a Deep Convolutional GAN (DCGAN) Architecture[1] and analyze how effectively it can be used to capture and replicate audio's sophisticated temporal and spectral relationships using multi channel image representations.

This model aims to focus on generating a category of audio and wholicsticly learn it's charactertics.

## 3. Data Manipulation

### 3.1. Collection

Training data is primarily sourced from digital production “sample packs.” For kick drums, the main "case study" for this paper, the training data used is a compilation of 7856 kick drum impules with different characteristics and use cases (analog, electronic, pop, hip-hop, beatbox, heavy, punchy, etc), overall providing a diverse range of potential drum sounds to generate that. A metric to watch for model validaiton is how well the model is able to generate the following set of "defining" kick drum characteristics.

A kick drum's "defining" characteristics include:

1. A transient: The “click” at the beginning of the generated audio incorporating most of the frequency spectrum
2. A fundamental: The sustained, decaying low frequency "rumble" after the transient
3. An overall "decaying" nature (spectral centroid shifts downwards)
4. Ample variability between decay times for each sample

<img alt='Features of a Kick Drum' src="static/kick-drum-features.png" width="350">
<p><b>Fig 1:</b> <i>Visualization of key features of a kick drum.</i></p>

### 3.2. Feature Extraction/Encoding

specifies audio shape then finds ideal hop length and frame size. Then cut data shape down to remove edge rtifact at end of sample (it egenrates slightly bigger than desired shape and includes an artifact only on those frames so thsi fixes both problems)

basically like yeah rewrite/modify this with the changes that fixed the fft artifact stuff YAY. fixing the vertical artifact thing is bigbrainnn

talk about istft time-freq mask, also how non stero data is duplicated then to be made stereo

Convolution by nature can not learn about the time-series component of audio data, thus this feature extraction process must flatten all the audio into a static form of data. This is achieved by representing audio with their magnitudes the time-frequency domain, similar to a spectrogram representation of audio. Each sample is first converted into a two channel array using a standard 44100 hz sampling rate. Then the audio sample is normalized to a length of 700 miliseconds and passed into a Short-time Fourier Transform (STFT). The STFT uses a kaiser window with a beta value of 14, a window size of of 512, and a hope size of 128. These parameters were determined to be the most effective through a signal reconstruction test (see STFT and iSTFT validation) and also limited by hardware contraints. The phase information of each frequency is then discarded and the data tensor's final shape is 2 channels by 245 frames by 257 frequency bins. This representation of information can be compared to that of a spectrogram.

While magnitude data (the output of the STFT) is important, this data is by nature skewed towards lower frequencies which contain higher intensities. To equalize the representation of frequencies in data the tensor of magnitude data is normalized to be in the range of 0 to 100 and then scaled into the logarithmic, decibal scale, which represents audio information as loudness, a more uniform scale relative to the entire frequency spectrum. The 0-100 scaling is to ensure consistent loudness information. A loudness threshold then sets all signals less than -90 decibals (auditory cutoff for the human ear) to be the minimum decibal value (a constant of -120) this data is then finally scaled to be between -1 and 1, representative of the output the model creates using the hyperbolic tangent activation function.

<img alt= 'Data processing comparison' src="static/magnitudes_vs_loudness.png" width=1000>
<p><b>Fig 2:</b> <i>STFT audio information before and after feature extraction.</i></p>

Generated audio representations apply the same audio scaling in the opposite direction. The audio is then reconstructed utilizing a inverse STFT in conjunction with the griffin-lim phase reconstruction algorithm[3]. This entire process preserves most, but not all audio information- see 5.2: STFT and iSTFT Validation.

## 4. Implementation

maybe reorg this whole section? talk abt generator, discriminator, then training loop
two foundational changes to making task happen with deep conv: discrimiator realizing its guessing and then adding attention, generator feature mathicng loss. a lot of other metrics help to smoothen stuff out but these two are main breakthrough things.

### 4.1. Architecture

This model seeks to replicate a DCGAN's multi-channel image generation capabilities[1] to create varied two channel audio representations.

### 4.2. Training

This work uses 80% of the dataset as training data and 20% as validation with all data split into batches of 16.

notes abt training losses and stuff:

mention that like for periodic penalty and stuff comparing real to fake instead of detemrining how periodic it is compared to like nothing should be smoother so it wont never make periodici only it was avoid sound qualities the real data does NOT have bc we dont want it to not be periodic bc model can misinterpret this kinda blanket rule so nudge it to be like the REAL data in terms of periodicity NOT like some kinda truly random noise just more closely macth real data- this is done with mostly all loss metrics THE IMPORTANT DISTINCTION TO TAKE NOTE OF FOR LOSS METRIC STUFF IS THAT THE LOSS FUNCTIONS ARE made to compare generated to real not generated to an arbitrary metric so it does that yeah

explain the like loss shaping stuff in detail brfhufrhufruhfruhfrhufr
the spectral rolloff function takes two spectrogram representation arrays, calculates which frequency bin at every frame it is where below that freqneyc bin is 85% of signal information, then return an array of all of these values
rolloff made it worse

if the centroid stuff doesnt give meaningful input idk how to make audio shape match better like have one speciifc kinda shape, maybe increase feature match/other existing metrics and also train longer?? idk

also: for the stuff like periodict pattern penalties talk abt how it initally was put into the discrim to indirectly control output but it had better effect in generator- if u end up putting in generator. if not then leave it as put in discrim to indirectly force it to avoid random periodic patterns. idek how to determinte the periodic stuff breh so implement this last tbh but also like idk

attention in the discriminator was genuinely lifechanign like real process syarted happening only then lmaoo like realizing it was guessing then fixing that is when all the discrim stuff started

FEATURE MATHCING IN THE GENERATOR IS ALSO SO LIFE CHANGING AHHAHAHA IT HELPS GENERATE MORE TONES AND MORE LENGTHS AHAHSHHADSHDSHDSAHDSA AND IT WORKS IT WORKS IT WORKS IT WORKS IT MAKES IT GENERATE DRUMS THE UNDERLYING TONES AND TEXTURES THERES A LOT OF STUFF STILL WRONG LIKE ARTIFACTS AND STUFF BUT ITS SO MUCH BETTER ITS USCH A GOOD METRIC ITS SUCH A FOUNDATIONAL BACKBONE METRIC

## 5. Results and Discussion

AUDIO TO TEST MODEL'S GENERATION CAAPABILITIES:

- kick
- snare/general perc foley stuff
- instrument shots(?)
- \>1sec some sound (idk what)

like in theory it can learn to create one phrase again and again and again but not tested bc data constraints, sample gan. genrated audio samples. StereoSampleGAN. developing my own model architecutre thing is actually insaneeee for like resume or apps or stuff tis a super super super cool result LOL!

Generate stereo instrumental audio (drums, snares/percs, instrument shots, etc- could work on vocal but like a one word thing adn not evaluated on spech generation), primarily use convolution/optmiize to work on low resource situations. moreso the whole point of it can geenretate speech but not in the tradiitonal sense of like its generating a word or phrase not text to speech or something, cant generate data of one single word of phrase to validate this but could be possible

so like find more data stuff and then like run through the model see how it works and if it is still okay-ish BOOM NEW MODEL ARCHITECTURE LFGGGGG but the provlem is it only generates 0.7sec audio so like impulseGAN or something?? like some kinda GAN based audio generator for impule based sounds or like quicker sounds that can be captued in 0.7 seconds or whatever like HELLL YEAHHHH THIS IS SO OP AND IT GENERATES STERO AUDIO TOO DUDEDEEEE THATS WHAT IT SETS APART BY, IMPULSEGAN GENEATES 0.7 SEC AUDIO IN STEREO THAT DOESNT HAVE TOBE IMPULSES SO COME UP WITH A BETTER NAEM IDK BUT THIS IS WHAT IM DOING, SO I GUESS IDK IFND LIKE DATA FOR SNARES OR WHATEVER LIKE AFTYER MODEL FULLY ENTIRELY WORKS FOR KICKS TEST QUICK WITH LIKE A SNARE LIBRARY AND SOME OTHER STUFF SEE IF IT CAN EFEFCTIVFELYT MAKE THOSE AND IF SO LIKE INCLUIDE THEM HERE IN DICSUSSION DUDE THIS IS AWESEOMEEEEE LETS GOOOO this is genuniely such a cool result dude WHATTTTTTT.

### 5.1. Model Evaluation

make one of those fire graphics of like 100 generated kick spectrograms and 100 data points the like 100 graphs in a box maybe not 100 like 20 stack both channels directly above each other and like create all 25 or whatever and put real data on left generated data on right but yeah this is a hella cool grsphic- can go in the README too!!!!

show learned kernels mayube??

### 5.2. Audio Specific Evaluations

oh drums do this, synths do this, snares do this, >1s audio does this, etc

### 5.3. STFT and iSTFT Validation

talk about audio processing valudation file stuff. tested rtisi, gla, gla w/ env matching, gla w/ time-freq masking, gla w magnitude guided reconstriuction, gla w/ freq bin weighting- after lots of experimentation found that gla with time freq masking produced the closest result to original data

explain signal reconstruction test + findings

talk abt auditory test comparing random examples w/ different windows, reconstruction algos, found rtisi better than griffin lim, wtv

important limitation of this method of audio generation

### 5.4. Contributions

overall learned info from this research:

model can potentially be used to train on other forms of data, propose a new conv based audio generation arhcitecture

specify that like all hte example testing stuff was with audio 0.5-1 seconds BUTTTTTT this architecture has the capability to generate longer audio but it isnt tested, likely would need an increase of image resolution (output from fft) which surpasses my hardware constraints dude this is big this is big this is big the whole like new gan approach to genrating stereo audio and ALSO the ability to have sounds LONGER THAN ONE SECOND DUDE THIS IS MASSSIVE ASGAGAGAGGHAHAGHAGAHG
probably specify tho that that kike best range of audio length will be sth like 2-5sec but it can def generate more would need to mess with fourier to adjust shape of data & like fourier paramaters to create more high quality info

## 6. Conclusion

## 7. References

<a id="1">[1]</a> CNN based GAN
https://arxiv.org/abs/1511.06434

<a id="2">[2]</a> GAN audio generation (WaveGAN)
https://arxiv.org/abs/1802.04208

<a id="3">[3]</a> Griffin Lim
https://speechprocessingbook.aalto.fi/Modelling/griffinlim.html
