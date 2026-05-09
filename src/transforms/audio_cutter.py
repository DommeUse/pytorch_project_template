import torch
import torch.nn.functional as F

class AudioCutter:
    def __init__(self, target_sr = 16000, duration = 0.5, is_train = True):
        self.target_size = int(target_sr * duration)
        self.is_train = is_train

    def __call__(self, audio):
        cur_size = audio.shape[-1]

        if cur_size < self.target_size:
            pad_size = self.target_size - cur_size
            return F.pad(audio, (0, pad_size), mode = 'replicate')
        
        if self.is_train:
            start = torch.randint(0, cur_size - self.target_size + 1, (1,)).item()
        else:
            start = (cur_size - self.target_size) // 2

        return audio[:, start : start + self.target_size]