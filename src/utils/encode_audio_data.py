from helpers import (
    audio_data_dir,
    average_spectrogram_path,
    compiled_data_path,
    compute_average_spectrogram,
    encode_sample_directory,
    graph_spectrogram,
    load_npy_data,
    normalized_db_to_wav,
)

# Encode samples
encode_sample_directory(audio_data_dir, silent=True)
compute_average_spectrogram()

real_data = load_npy_data(compiled_data_path)  # datapts, channels, frames, freq bins
average_data = load_npy_data(average_spectrogram_path)  # channels, frames, freq bins
print("Data " + str(real_data.shape))
print("Average " + str(average_data.shape))

graph_spectrogram(average_data, "Average Data Point")
