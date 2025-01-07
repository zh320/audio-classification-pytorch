import torchaudio
import torchaudio.transforms as T


def preprocess_audio(file_path, sample_rate=44100):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate)(waveform)
    mel_spectrogram = T.AmplitudeToDB(stype="power", top_db=80)(mel_spectrogram)

    return mel_spectrogram