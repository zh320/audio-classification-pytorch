import os
import pandas as pd
from torch.utils.data import Dataset

from .transforms import preprocess_audio


class ESC50(Dataset):
    def __init__(self, config, mode='train'):
        assert mode in ['train', 'val']
        assert config.val_fold >= 1 and config.val_fold <= 5, "`config.val_fold` must be between 1 and 5."
        data_root = os.path.expanduser(config.data_root)
        self.audio_dir = os.path.join(data_root, 'audio')
        meta_dir = os.path.join(data_root, 'meta')
        self.sample_rate = config.sample_rate

        if not os.path.isdir(self.audio_dir):
            raise RuntimeError(f'Audio directory: {self.audio_dir} does not exist.')

        if not os.path.isdir(meta_dir):
            raise RuntimeError(f'Meta directory: {meta_dir} does not exist.')

        metadata = pd.read_csv(f'{meta_dir}/esc50.csv')
        select_fold = metadata['fold'] != config.val_fold if mode == 'train' else metadata['fold'] == config.val_fold
        self.metadata = metadata[select_fold].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        file_path = f"{self.audio_dir}/{row['filename']}"
        label = row['target']

        mel_spectrogram = preprocess_audio(file_path, sample_rate=self.sample_rate)

        return mel_spectrogram, label