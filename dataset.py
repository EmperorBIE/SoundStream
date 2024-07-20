from torch.utils.data import Dataset
import torchaudio

import glob
import json

class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir):
        super().__init__()
        
        self.filenames = glob.glob(audio_dir+"/*.wav")
        _, self.sr = torchaudio.load(self.filenames[0])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        return torchaudio.load(self.filenames[index])[0]


class JsonDataset(Dataset):
    """Dataset to load NSynth data from a JSON file."""

    def __init__(self, json_dir):
        super().__init__()
        
        with open(json_dir, 'r') as f:
            self.data_info = json.load(f)
        
        self.filenames = [item['file_path'] for item in self.data_info.values()]
        
        if self.filenames:
            _, self.sr = torchaudio.load(self.filenames[0])
        else:
            self.sr = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return torchaudio.load(self.filenames[index])[0]