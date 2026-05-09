from src.datasets.librispeech_dataset import LibriSpeechDataset

ds = LibriSpeechDataset(part = "test-clean")
print(len(ds))

audio = ds[13]
print("Shape:", audio['audio'].shape)
print("Audio length:", audio['audio_len'])