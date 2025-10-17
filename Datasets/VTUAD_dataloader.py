import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import pytorch_lightning as pl


class VTUADSegments(Dataset):
    """
    Dataset for VTUAD with the structure:
    combined_scenario/
      train/
        audio/
          background/*.wav
          cargo/*.wav
          passengership/*.wav
          tanker/*.wav
          tug/*.wav
      validation/
        audio/<same-5-classes>/*.wav
      test/
        audio/<same-5-classes>/*.wav
    """
    def __init__(self, root_dir, split='train', transform=None, target_transform=None,
                 norm_function=None, class_mapping=None, strict_checks=True):
        """
        Args:
            root_dir (str): Path to 'combined_scenario'.
            split (str): One of {'train','validation','test'}.
            transform: Optional callable for waveform.
            target_transform: Optional callable for label tensor.
            norm_function: Optional callable for normalization (set later from DataModule).
            class_mapping (dict): Optional explicit class->index mapping.
            strict_checks (bool): If True, raise if class folders are missing.
        """
        assert split in {'train', 'validation', 'test'}
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.norm_function = norm_function

        # Finalized mapping (fixed duplicate id in user text)
        self.class_mapping = class_mapping or {
            'background': 0,
            'cargo': 1,
            'passengership': 2,
            'tanker': 3,
            'tug': 4
        }

        self.split_audio_dir = os.path.join(self.root_dir, split, 'audio')
        if not os.path.isdir(self.split_audio_dir):
            raise FileNotFoundError(f"Expected split folder: {self.split_audio_dir}")

        # Collect files
        self.items = []
        missing_classes = []
        for class_name, class_idx in self.class_mapping.items():
            cdir = os.path.join(self.split_audio_dir, class_name)
            if not os.path.isdir(cdir):
                missing_classes.append(class_name)
                if strict_checks:
                    raise FileNotFoundError(f"Missing class folder: {cdir}")
                else:
                    continue

            for root, _, files in os.walk(cdir):
                for f in files:
                    if f.lower().endswith('.wav'):
                        self.items.append((os.path.join(root, f), class_idx))

        if not self.items:
            raise RuntimeError(f"No .wav files found under {self.split_audio_dir}")

        # Optional: quick tally per class (helpful for debugging)
        # counts = {k: 0 for k in self.class_mapping.values()}
        # for _, y in self.items:
        #     counts[y] += 1
        # print(f"[VTUAD] {split} counts:", counts)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, label = self.items[idx]
        try:
            sr, signal = wavfile.read(fp, mmap=False)
        except Exception as e:
            raise RuntimeError(f"Error reading file {fp}: {e}")

        # Convert to float32
        signal = signal.astype(np.float32)

        # Normalize if provided
        if self.norm_function is not None:
            signal = self.norm_function(signal)

        # Optional transform on waveform
        if self.transform is not None:
            signal = self.transform(signal)

        x = torch.tensor(signal)
        if torch.isnan(x).any():
            raise ValueError(f"NaN values found in signal from file {fp}")

        y = torch.tensor(label, dtype=torch.long)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y, idx

    def compute_global_min_max(self):
        """Compute min/max across this split (typically called on the train split)."""
        gmin = float('inf')
        gmax = float('-inf')
        for fp, _ in self.items:
            try:
                _, sig = wavfile.read(fp, mmap=False)
                sig = sig.astype(np.float32)
                smin, smax = float(np.min(sig)), float(np.max(sig))
                if smin < gmin:
                    gmin = smin
                if smax > gmax:
                    gmax = smax
            except Exception as e:
                raise RuntimeError(f"Error reading file {fp}: {e}")
        return gmin, gmax

    def set_norm_function(self, global_min, global_max):
        eps = 1e-12
        scale = (global_max - global_min)
        if scale < eps:
            # Avoid divide-by-zero; fall back to zero-centering
            self.norm_function = lambda x: x - global_min
            print(f"[VTUAD:{self.split}] Degenerate range; using zero-centering with min={global_min:.4f}")
        else:
            self.norm_function = lambda x: (x - global_min) / scale
            print(f"[VTUAD:{self.split}] Normalization set (min={global_min:.4f}, max={global_max:.4f})")

    def count_classes(self):
        counts = {k: 0 for k in self.class_mapping.values()}
        for _, y in self.items:
            counts[y] += 1
        # back to names
        inv_map = {v: k for k, v in self.class_mapping.items()}
        return {inv_map[i]: n for i, n in counts.items()}


class VTUADDataModule(pl.LightningDataModule):
    """
    Lightning DataModule that uses the fixed VTUAD splits (no random splitting).
    Normalization is computed from the TRAIN split and applied to val/test.
    """
    def __init__(self, data_dir, batch_size, num_workers=4, pin_memory=True,
                 class_mapping=None, persistent_workers=True):
        """
        Args:
            data_dir (str): Path to 'combined_scenario'.
            batch_size (dict): e.g., {'train': 64, 'val': 64, 'test': 64}
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.class_mapping = class_mapping or {
            'background': 0,
            'cargo': 1,
            'passengership': 2,
            'tanker': 3,
            'tug': 4
        }

        self.global_min = None
        self.global_max = None
        self.norm_function = None

    def prepare_data(self):
        # Nothing to download—data already organized on disk.
        pass

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_dataset = VTUADSegments(
                root_dir=self.data_dir, split='train', class_mapping=self.class_mapping
            )
            self.val_dataset = VTUADSegments(
                root_dir=self.data_dir, split='validation', class_mapping=self.class_mapping
            )

            # Compute normalization from TRAIN only
            self.global_min, self.global_max = self.train_dataset.compute_global_min_max()
            self.train_dataset.set_norm_function(self.global_min, self.global_max)
            self.val_dataset.set_norm_function(self.global_min, self.global_max)

        if stage in (None, 'test'):
            self.test_dataset = VTUADSegments(
                root_dir=self.data_dir, split='test', class_mapping=self.class_mapping
            )
            if self.global_min is None or self.global_max is None:
                raise RuntimeError("Fit stage must run before test to compute normalization.")
            self.test_dataset.set_norm_function(self.global_min, self.global_max)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size['train'],
            shuffle=True, num_workers=self.num_workers,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size['val'],
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size['test'],
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
