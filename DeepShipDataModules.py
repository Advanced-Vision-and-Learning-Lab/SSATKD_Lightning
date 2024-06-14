import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import pytorch_lightning as pl

class DeepShipSegments(Dataset):
    def __init__(self, parent_folder, train_split=0.7, val_test_split=0.5,
                 partition='train', random_seed=42, shuffle=False, transform=None, 
                 target_transform=None):
        self.parent_folder = parent_folder
        self.folder_lists = {'train': [], 'test': [], 'val': []}
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.partition = partition
        self.transform = transform
        self.shuffle = shuffle
        self.target_transform = target_transform
        self.random_seed = random_seed
        self.norm_function = None
        self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        self._prepare_data()

    def _prepare_data(self):
        for label in self.class_mapping.keys():
            label_path = os.path.join(self.parent_folder, label)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Directory {label_path} does not exist.")
            subfolders = os.listdir(label_path)
            subfolders_train, subfolders_test_val = train_test_split(
                subfolders, train_size=self.train_split, shuffle=self.shuffle, random_state=self.random_seed
            )
            subfolders_test, subfolders_val = train_test_split(
                subfolders_test_val, train_size=self.val_test_split, shuffle=self.shuffle, random_state=self.random_seed
            )
            self._add_subfolders_to_list(label_path, subfolders_train, 'train', label)
            self._add_subfolders_to_list(label_path, subfolders_test, 'test', label)
            self._add_subfolders_to_list(label_path, subfolders_val, 'val', label)

        self.segment_lists = {'train': [], 'test': [], 'val': []}
        self._collect_segments()

    def _add_subfolders_to_list(self, label_path, subfolders, split, label):
        for subfolder in subfolders:
            subfolder_path = os.path.join(label_path, subfolder)
            self.folder_lists[split].append((subfolder_path, self.class_mapping[label]))

    def _collect_segments(self):
        for split in self.folder_lists.keys():
            for folder, label in self.folder_lists[split]:
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith('.wav'):
                            file_path = os.path.join(root, file)
                            self.segment_lists[split].append((file_path, label))

    def __len__(self):
        return len(self.segment_lists[self.partition])

    def __getitem__(self, idx):
        file_path, label = self.segment_lists[self.partition][idx]    
        try:
            sr, signal = wavfile.read(file_path, mmap=False)
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")

        signal = signal.astype(np.float32)
        if self.norm_function is not None:
            signal = self.norm_function(signal)
        signal = torch.tensor(signal)

        label = torch.tensor(label)
        if self.target_transform:
            label = self.target_transform(label)

        return signal, label, idx

class DeepShipDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, pin_memory, train_split=0.7, val_test_split=0.5, random_seed=42, shuffle=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.norm_function = None

    def prepare_data(self):
        # Download or prepare data if necessary
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DeepShipSegments(self.data_dir, partition='train', train_split=self.train_split, val_test_split=self.val_test_split, random_seed=self.random_seed, shuffle=self.shuffle)
            self.val_dataset = DeepShipSegments(self.data_dir, partition='val', train_split=self.train_split, val_test_split=self.val_test_split, random_seed=self.random_seed, shuffle=self.shuffle)
            self.norm_function = self._get_normalization_function(self.train_dataset)
            self.train_dataset.norm_function = self.norm_function
            self.val_dataset.norm_function = self.norm_function

        if stage == 'test' or stage is None:
            self.test_dataset = DeepShipSegments(self.data_dir, partition='test', train_split=self.train_split, val_test_split=self.val_test_split, random_seed=self.random_seed, shuffle=self.shuffle)
            self.test_dataset.norm_function = self.norm_function

    def _get_normalization_function(self, dataset):
        # Placeholder function to compute normalization function from dataset
        return lambda x: (x - x.min()) / (x.max() - x.min())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size['train'], shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size['val'], shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size['test'], shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)