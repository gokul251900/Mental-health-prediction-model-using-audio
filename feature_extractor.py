import librosa
import numpy as np

def extract_features(audio_path):

    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc, axis=1)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    energy = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)
    duration = librosa.get_duration(y=y, sr=sr)

    speech_rate = len(y) / duration

    pause_frequency = np.sum(energy < 0.01) / len(energy)

    features = np.concatenate([
        mfcc,
        [pitch_mean, pitch_std, energy_mean, energy_std, speech_rate, pause_frequency]
    ])

    return features