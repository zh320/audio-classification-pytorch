import os
from torch.utils.data import Dataset

from .transforms import preprocess_audio


class TestDataset(Dataset):
    def __init__(self, config):
        data_folder = os.path.expanduser(config.test_data_folder)
        self.sample_rate = config.sample_rate

        if not os.path.isdir(data_folder):
            raise RuntimeError(f'Test directory: {data_folder} does not exist.')

        self.audios = []
        self.aud_names = []

        for file_name in os.listdir(data_folder):
            self.audios.append(os.path.join(data_folder, file_name))
            self.aud_names.append(file_name)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        aud_name = self.aud_names[index]

        mel_spectrogram = preprocess_audio(self.audios[index], sample_rate=self.sample_rate)

        return mel_spectrogram, aud_name
