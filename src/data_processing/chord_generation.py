import numpy as np
from scipy.io import wavfile
import os
from itertools import product
import random
from typing import Dict, List, Callable, Optional


class WaveformGenerator:
    @staticmethod
    def sine(t: np.ndarray, freq: float) -> np.ndarray:
        """Generate sine waveform."""
        return np.sin(2 * np.pi * freq * t)

    @staticmethod
    def square(t: np.ndarray, freq: float) -> np.ndarray:
        """Generate square waveform."""
        return np.sign(np.sin(2 * np.pi * freq * t))

    @staticmethod
    def sawtooth(t: np.ndarray, freq: float) -> np.ndarray:
        """Generate sawtooth waveform."""
        return 2 * (freq * t - np.floor(0.5 + freq * t))

    @staticmethod
    def triangle(t: np.ndarray, freq: float) -> np.ndarray:
        """Generate triangle waveform."""
        return 2 * np.abs(2 * (freq * t - np.floor(0.5 + freq * t))) - 1


class ChordGenerator:
    def __init__(self, sample_rate: int = 44100, duration: float = 1.5):
        self.sample_rate = sample_rate
        self.duration = duration

        # Define waveforms
        self.waveforms: Dict[str, Callable] = {
            "sine": WaveformGenerator.sine,
            "square": WaveformGenerator.square,
            "sawtooth": WaveformGenerator.sawtooth,
            "triangle": WaveformGenerator.triangle,
        }

        # Chord ratios
        self.chord_types: Dict[str, List[float]] = {
            "minor": [1, 1.2, 1.5],  # Minor triad
            "major": [1, 1.25, 1.5],  # Major triad
            "diminished": [1, 1.189, 1.414],  # Diminished triad
            "augmented": [1, 1.25, 1.587],  # Augmented triad
            "minor7": [1, 1.2, 1.5, 1.782],  # Minor seventh
            "major7": [1, 1.25, 1.5, 1.87],  # Major seventh
            "dom7": [1, 1.25, 1.5, 1.782],  # Dominant seventh
            "sus2": [1, 1.122, 1.5],  # Suspended second
            "sus4": [1, 1.335, 1.5],  # Suspended fourth
        }

        # Note voicings
        self.voicings: Dict[str, Callable] = {
            "normal": lambda n: np.ones(n),  # Equal volume for all notes
            "root_heavy": lambda n: np.linspace(1.2, 0.8, n),  # Emphasize root note
            "top_heavy": lambda n: np.linspace(0.8, 1.2, n),  # Emphasize highest note
        }

        # Noise types
        self.noise_types: Dict[str, Callable] = {
            "white": lambda size: np.random.normal(0, 1, size),
            "pink": self.generate_pink_noise,
            "brown": self.generate_brown_noise,
            "none": lambda size: np.zeros(size),
        }

    def generate_adsr_envelope(
        self,
        size: int,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
    ) -> np.ndarray:
        """Generate ADSR envelope with given parameters and timing offsets."""
        t = np.linspace(0, 1, size)

        # Calculate timing with offsets
        effective_start = max(0.0, start_offset)
        effective_end = min(1.0, 1.0 - end_offset)

        # Adjust envelope timings
        attack_idx = int(
            (effective_start + attack * (effective_end - effective_start)) * size
        )
        decay_idx = int(
            (effective_start + (attack + decay) * (effective_end - effective_start))
            * size
        )
        release_idx = int(effective_end * size)

        envelope = np.zeros_like(t)

        # Apply start offset
        if effective_start > 0:
            envelope[: int(effective_start * size)] = 0

        # Generate envelope segments
        if attack_idx > int(effective_start * size):
            envelope[int(effective_start * size) : attack_idx] = np.linspace(
                0, 1, attack_idx - int(effective_start * size)
            )

        if decay_idx > attack_idx:
            envelope[attack_idx:decay_idx] = np.linspace(
                1, sustain, decay_idx - attack_idx
            )

        envelope[decay_idx:release_idx] = sustain

        # Apply release
        if release_idx < size:
            release_samples = size - release_idx
            envelope[release_idx:] = np.linspace(sustain, 0, release_samples)

        return envelope

    def generate_tone(
        self,
        frequency: float,
        waveform: str = "sine",
        adsr_params: Optional[Dict] = None,
        amplitude: float = 0.5,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
    ) -> np.ndarray:
        """Generate a frequency tone with waveform, ADSR envelope, and timing variations."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)

        # Generate the basic waveform
        if waveform == "pulse":
            duty = random.uniform(0.1, 0.9)
            wave = self.waveforms[waveform](t, frequency, duty)
        else:
            wave = self.waveforms[waveform](t, frequency)

        # Apply ADSR envelope with timing variations
        if adsr_params is None:
            adsr_params = {
                "attack": random.uniform(0.05, 0.2),
                "decay": random.uniform(0.1, 0.3),
                "sustain": random.uniform(0.4, 0.8),
                "release": random.uniform(0.1, 0.3),
            }

        envelope = self.generate_adsr_envelope(
            len(t),
            adsr_params["attack"],
            adsr_params["decay"],
            adsr_params["sustain"],
            adsr_params["release"],
            start_offset,
            end_offset,
        )

        return amplitude * wave * envelope

    def generate_pink_noise(self, size: int) -> np.ndarray:
        """Generate pink noise."""
        f = np.fft.fftfreq(size)
        f[0] = float("inf")
        spectrum = 1 / np.abs(f)
        spectrum[0] = 0
        phase = np.random.uniform(0, 2 * np.pi, size)
        noise = np.fft.ifft(spectrum * np.exp(1j * phase)).real
        return noise / np.std(noise)

    def generate_brown_noise(self, size: int) -> np.ndarray:
        """Generate brown noise."""
        noise = np.cumsum(np.random.normal(0, 1, size))
        return noise / np.std(noise)

    def generate_chord(
        self,
        root_freq: float,
        chord_type: str = "minor",
        voicing: str = "normal",
        waveform: str = "sine",
        noise_type: str = "white",
        noise_level: float = 0.02,
        timing_variation: float = 0.02,
    ) -> np.ndarray:
        """Generate a chord with specified parameters, random ADSR, and timing variations."""
        intervals = self.chord_types[chord_type]
        weights = self.voicings[voicing](len(intervals))

        # Generate random ADSR parameters
        adsr_params = {
            "attack": random.uniform(0.05, 0.2),
            "decay": random.uniform(0.1, 0.3),
            "sustain": random.uniform(0.4, 0.8),
            "release": random.uniform(0.1, 0.3),
        }

        # Generate tones
        chord = np.zeros(int(self.sample_rate * self.duration))

        for w, interval in zip(weights, intervals):
            # Generate random timing offsets for each note
            start_offset = random.uniform(0, timing_variation)
            end_offset = random.uniform(0, timing_variation)

            note = w * self.generate_tone(
                root_freq * interval,
                waveform,
                adsr_params,
                start_offset=start_offset,
                end_offset=end_offset,
            )
            chord += note

        # Add noise
        if noise_type != "none":
            noise = self.noise_types[noise_type](len(chord)) * noise_level
            chord = chord + noise

        # Normalize
        max_val = np.max(np.abs(chord))
        if max_val > 0:
            chord = chord / max_val

        return chord


def generate_dataset(output_dir: str = "data/chordshot_samples", n_samples: int = 2):
    """Generate a large dataset of chord samples."""
    generator = ChordGenerator()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parameters for variation
    base_freqs = np.logspace(np.log10(130.81), np.log10(523.25), 25)  # C2 to C5
    noise_levels = [0.01, 0.02, 0.03]
    timing_variations = [0.01, 0.02, 0.03]

    # Create parameter combinations
    generated_sample_count = 0
    param_combinations = list(
        product(
            base_freqs,
            generator.chord_types.keys(),
            generator.voicings.keys(),
            generator.waveforms.keys(),
            generator.noise_types.keys(),
            noise_levels,
            timing_variations,
        )
    )
    random.shuffle(param_combinations)

    # Create samples
    for (
        freq,
        chord_type,
        voicing,
        waveform,
        noise_type,
        noise_level,
        timing_variation,
    ) in param_combinations:
        if generated_sample_count >= n_samples:
            break

        # Generate the chord
        try:
            chord = generator.generate_chord(
                root_freq=float(freq),
                chord_type=str(chord_type),
                voicing=str(voicing),
                waveform=str(waveform),
                noise_type=str(noise_type),
                noise_level=float(noise_level),
                timing_variation=float(timing_variation),
            )

            # Create filename and save
            filename = f"chord_{(generated_sample_count + 1):04d}.wav"
            filepath = os.path.join(output_dir, filename)
            wavfile.write(
                filepath, generator.sample_rate, (chord * 32767).astype(np.int16)
            )

            generated_sample_count += 1

        except Exception as e:
            print(f"Error generating sample: {e}")
            continue

    print(f"Generated {generated_sample_count} chord samples in {output_dir}")


generate_dataset(n_samples=4000)
