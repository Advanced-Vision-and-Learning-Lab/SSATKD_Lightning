#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 12:10:32 2025

@author: jarin.ritu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UrbanSound8K DataModule
- Train/val/test by folds (defaults: train=1-8, val=9, test=10)
- Loads raw audio, resamples, mono, pads/trims to fixed duration
- Computes global (train) min/max for normalization and applies to val/test
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pytorch_lightning as pl


# ---------------------------
# Dataset
# ---------------------------
class UrbanSound8KDataset(Dataset):
    """
    Returns: (signal_1d_float32, label_int64, idx)
    - signal is mono, length = sample_rate * duration_sec (padded/trimmed)
    - label is classID from metadata
    """

    def __init__(
        self,
        root: str,
        folds: List[int],
        sample_rate: int = 16000,
        duration_sec: float = 4.0,
        shuffle: bool = True,
        random_seed: int = 42,
        waveform_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        norm_function: Optional[Callable] = None,
    ):
        """
        Args:
            root: path to UrbanSound8K directory with subfolders:
                  - audio/fold1 ... fold10
                  - metadata/UrbanSound8K.csv
            folds: which folds to include in this split
            sample_rate: target sampling rate
            duration_sec: fixed output duration (pad/trim)
            waveform_transform: optional transform on waveform tensor
            target_transform: optional transform on target label
            norm_function: normalization fn taking np.ndarray -> np.ndarray
        """
        super().__init__()
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.target_len = int(sample_rate * duration_sec)

        self.waveform_transform = waveform_transform
        self.target_transform = target_transform
        self.norm_function = norm_function

        self.shuffle = shuffle
        self.random_seed = random_seed

        meta_path = self.root / "metadata" / "UrbanSound8K.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        df = pd.read_csv(meta_path)
        if not {"slice_file_name", "fold", "classID"}.issubset(df.columns):
            raise ValueError("UrbanSound8K.csv missing required columns")

        df = df[df["fold"].isin(folds)].copy()
        if df.empty:
            raise ValueError(f"No files found for folds: {folds}")

        # Build a list of (absolute_path, classID)
        items: List[Tuple[str, int]] = []
        audio_root = self.root / "audio"
        for _, row in df.iterrows():
            f = int(row["fold"])
            fname = row["slice_file_name"]
            class_id = int(row["classID"])
            wav_path = audio_root / f"fold{f}" / fname
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing file: {wav_path}")
            items.append((str(wav_path), class_id))

        if shuffle:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(items)

        self.items = items

        # Prepare torchaudio ops
        self.resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}
        self._ensure_resampler(self.sample_rate)  # Pre-create for expected SR

    def _ensure_resampler(self, orig_sr: int):
        if orig_sr not in self.resampler_cache:
            self.resampler_cache[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )

    def __len__(self):
        return len(self.items)

    def _load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        # waveform: (channels, n_samples), sr: int
        waveform, sr = torchaudio.load(path)
        # to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # resample if needed
        if sr != self.sample_rate:
            self._ensure_resampler(sr)
            waveform = self.resampler_cache[sr](waveform)
            sr = self.sample_rate
        return waveform, sr

    def _pad_or_trim(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (1, n)
        n = wav.shape[-1]
        if n == self.target_len:
            return wav
        elif n > self.target_len:
            return wav[..., : self.target_len]
        else:
            # pad at end with zeros
            pad_len = self.target_len - n
            return torch.nn.functional.pad(wav, (0, pad_len))

    def compute_global_min_max(self) -> Tuple[float, float]:
        gmin = float("inf")
        gmax = float("-inf")
        for p, _ in self.items:
            wav, sr = self._load_audio(p)  # (1, n)
            arr = wav.squeeze(0).numpy().astype(np.float32)
            # pad/trim similarly to ensure consistent range
            if arr.shape[-1] != self.target_len:
                # quick pad/trim in numpy
                if arr.shape[-1] > self.target_len:
                    arr = arr[: self.target_len]
                else:
                    arr = np.pad(arr, (0, self.target_len - arr.shape[-1]))
            gmin = min(gmin, float(np.min(arr)))
            gmax = max(gmax, float(np.max(arr)))
        return gmin, gmax

    def set_norm_function(self, global_min: float, global_max: float):
        denom = (global_max - global_min) if (global_max > global_min) else 1.0
        self.norm_function = lambda x: (x - global_min) / denom
        print(f"[UrbanSound8KDataset] Norm set: global_min={global_min:.6f}, global_max={global_max:.6f}")

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        wav, sr = self._load_audio(path)            # (1, n)
        wav = self._pad_or_trim(wav).squeeze(0)     # (n,) float32

        # numpy normalize if requested (matches your VTUAD pattern)
        if self.norm_function is not None:
            arr = wav.numpy()
            arr = self.norm_function(arr).astype(np.float32)
            wav = torch.from_numpy(arr)

        # optional waveform transform (expects torch.Tensor)
        if self.waveform_transform is not None:
            wav = self.waveform_transform(wav)

        y = torch.tensor(label, dtype=torch.long)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return wav, y, idx


# ---------------------------
# DataModule
# ---------------------------
class UrbanSound8KDataModule(pl.LightningDataModule):
    """
    Usage:
        dm = UrbanSound8KDataModule(
            data_dir="/path/UrbanSound8K",
            batch_size={"train": 64, "val": 64, "test": 64},
            sample_rate=16000,
            duration_sec=4.0,
            train_folds=list(range(1,9)),
            val_folds=[9],
            test_folds=[10],
            num_workers=4,
            pin_memory=True,
        )
        dm.prepare_data()
        dm.setup('fit')
        loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: Dict[str, int],
        sample_rate: int = 16000,
        duration_sec: float = 4.0,
        train_folds: Optional[List[int]] = None,
        val_folds: Optional[List[int]] = None,
        test_folds: Optional[List[int]] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        random_seed: int = 42,
        waveform_transform_train: Optional[Callable] = None,
        waveform_transform_eval: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.waveform_transform_train = waveform_transform_train
        self.waveform_transform_eval = waveform_transform_eval
        self.target_transform = target_transform

        self.train_folds = train_folds if train_folds is not None else list(range(1, 9))
        self.val_folds = val_folds if val_folds is not None else [9]
        self.test_folds = test_folds if test_folds is not None else [10]

        self.global_min: Optional[float] = None
        self.global_max: Optional[float] = None

        self.train_dataset: Optional[UrbanSound8KDataset] = None
        self.val_dataset: Optional[UrbanSound8KDataset] = None
        self.test_dataset: Optional[UrbanSound8KDataset] = None

    def prepare_data(self):
        # nothing to download; verify structure exists
        meta = Path(self.data_dir) / "metadata" / "UrbanSound8K.csv"
        if not meta.exists():
            raise FileNotFoundError(f"Expected metadata at {meta}")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = UrbanSound8KDataset(
                root=self.data_dir,
                folds=self.train_folds,
                sample_rate=self.sample_rate,
                duration_sec=self.duration_sec,
                shuffle=self.shuffle,
                random_seed=self.random_seed,
                waveform_transform=self.waveform_transform_train,
                target_transform=self.target_transform,
                norm_function=None,  # set after computing global min-max
            )
            self.val_dataset = UrbanSound8KDataset(
                root=self.data_dir,
                folds=self.val_folds,
                sample_rate=self.sample_rate,
                duration_sec=self.duration_sec,
                shuffle=False,
                waveform_transform=self.waveform_transform_eval,
                target_transform=self.target_transform,
                norm_function=None,
            )

            # compute global min/max on TRAIN, then apply to train/val
            self.global_min, self.global_max = self.train_dataset.compute_global_min_max()
            self.train_dataset.set_norm_function(self.global_min, self.global_max)
            self.val_dataset.set_norm_function(self.global_min, self.global_max)

        if stage == "test" or stage is None:
            self.test_dataset = UrbanSound8KDataset(
                root=self.data_dir,
                folds=self.test_folds,
                sample_rate=self.sample_rate,
                duration_sec=self.duration_sec,
                shuffle=False,
                waveform_transform=self.waveform_transform_eval,
                target_transform=self.target_transform,
                norm_function=None,
            )
            # require fit to have run to get global stats
            if self.global_min is None or self.global_max is None:
                # In case user calls setup('test') first, recompute from train folds here
                tmp_train = UrbanSound8KDataset(
                    root=self.data_dir,
                    folds=self.train_folds,
                    sample_rate=self.sample_rate,
                    duration_sec=self.duration_sec,
                    shuffle=False,
                )
                self.global_min, self.global_max = tmp_train.compute_global_min_max()
            self.test_dataset.set_norm_function(self.global_min, self.global_max)

    # Dataloaders
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
