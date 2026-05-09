import json
import os
import shutil
from pathlib import Path

import soundfile as sf
import torchaudio
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
}

class LibriSpeechDataset(BaseDataset):
    def __init__(self, part, data_dir = None, *args, **kwargs):
        assert part in URL_LINKS, ("Dataset part name is incorrect! Try to change it.")

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok = True, parents = True)

        self._data_dir = Path(data_dir)
        
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        print(f"Loading split {part}")
        archive_path = self._data_dir / f"{part}.tar.gz"

        wget.download(URL_LINKS[part], str(archive_path))
        shutil.unpack_archive(archive_path, self._data_dir)
        shutil.move(str(self._data_dir / "LibriSpeech" / part), str(self._data_dir / part))
        os.remove(str(archive_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _create_index(self, part):
        index = []
        split_path = self._data_dir / part

        if not split_path.exists():
            self._load_part(part)
        
        for flac_path in tqdm(list(split_path.rglob("*.flac")), desc = f"Preparing librispeech split {part}"):
            audio_info = sf.info(str(flac_path))
            index.append({'path' : str(flac_path), 'audio_len': audio_info.frames / audio_info.samplerate})

        return index
    
    def _get_or_load_index(self, part):
        index_path = ROOT_PATH / "data" / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            index_path.parent.mkdir(parents = True, exist_ok = True)
            with index_path.open("w") as f:
                json.dump(index, f, indent = 2)
        return index
