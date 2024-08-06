from utils.helpers import (
    amplitudes_to_wav,
    encode_sample,
    graph_spectrogram,
    scale_normalized_db_to_amplis,
)

sample = "/Users/abhayshukla/Documents/GitHub/deep-convolution-audio-generation/data/kick_samples/(OS) kick doorknocker.wav"
loudness = encode_sample(sample)

amplis = scale_normalized_db_to_amplis(loudness)
graph_spectrogram(amplis, "after scaling back to amplitudes")

amplitudes_to_wav(amplis, "test")

saved = "/Users/abhayshukla/Documents/GitHub/deep-convolution-audio-generation/model/test.wav"
loudness2 = encode_sample(saved)
