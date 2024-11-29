import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import pytorch_lightning as pl


class DeepShipSegments(Dataset):
    def __init__(self, parent_folder, train_split=0.7, val_split=0.2, test_split=0.1,
                 partition='train', random_seed=42, shuffle=True, transform=None, 
                 target_transform=None, norm_function=None):
        assert train_split + val_split + test_split == 1, "Splits must add up to 1"
        self.parent_folder = parent_folder
        self.folder_lists = {'train': [], 'test': [], 'val': []}
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.partition = partition
        self.transform = transform
        self.shuffle = shuffle
        self.target_transform = target_transform
        self.random_seed = random_seed
        self.norm_function = norm_function
        self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        self._prepare_data()
        self.global_min, self.global_max = self.compute_global_min_max()

    def _prepare_data(self):
        for label in self.class_mapping.keys():
            label_path = os.path.join(self.parent_folder, label)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Directory {label_path} does not exist.")
            subfolders = os.listdir(label_path)

            if len(subfolders) < 3:
                raise ValueError(f"Not enough subfolders to split for label '{label}'. Need at least 3.")
            
            num_samples = len(subfolders)
            train_size = int(num_samples * self.train_split)
            val_size = int(num_samples * self.val_split)
            test_size = num_samples - train_size - val_size
            
            if train_size + val_size + test_size > num_samples:
                raise ValueError(f"The sum of train_size, val_size, and test_size must be less than the number of samples ({num_samples}).")
            
            subfolders_train_val, subfolders_test = train_test_split(
                subfolders, train_size=train_size + val_size, 
                test_size=test_size, shuffle=self.shuffle, random_state=self.random_seed
            )
            
            val_size_adjusted = val_size / (train_size + val_size)
            subfolders_train, subfolders_val = train_test_split(
                subfolders_train_val, train_size=1 - val_size_adjusted, test_size=val_size_adjusted, 
                shuffle=self.shuffle, random_state=self.random_seed
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
            if self.shuffle:
                np.random.seed(self.random_seed)
                np.random.shuffle(self.segment_lists[split])
            # print(f"{split.capitalize()} set: {len(self.folder_lists[split])} folders, {len(self.segment_lists[split])} segments")
            label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for _, label in self.segment_lists[split]:
                label_counts[label] += 1
            # print(f"{split.capitalize()} label distribution: {label_counts}")
            if split == 'test':
                with open('test_folders.txt', 'w') as f:
                    f.write(f"\nFolders in the test set ({len(self.folder_lists['test'])} folders):\n")
                    for folder, label in self.folder_lists['test']:
                        f.write(f"Folder: {folder}, Label: {label}\n")
    def count_classes_per_split(self):
        # Initialize a dictionary to store counts for each split
        class_counts = {
            'train': {label: 0 for label in self.class_mapping.values()},
            'val': {label: 0 for label in self.class_mapping.values()},
            'test': {label: 0 for label in self.class_mapping.values()}
        }
        
        # Count occurrences of each class in the respective split
        for split in ['train', 'val', 'test']:
            for _, label in self.segment_lists[split]:
                class_counts[split][label] += 1
        
        # Convert numeric labels back to class names for readability
        class_names = {v: k for k, v in self.class_mapping.items()}
        for split in class_counts:
            class_counts[split] = {class_names[label]: count for label, count in class_counts[split].items()}
        
        return class_counts

    def __len__(self):
        return len(self.segment_lists[self.partition])

    def __getitem__(self, idx):
        file_path, label = self.segment_lists[self.partition][idx]
        # print("\nfile path",file_path)
        try:
            sr, signal = wavfile.read(file_path, mmap=False)
            # print("signal",signal.shape)
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")

        signal = signal.astype(np.float32)
        if self.norm_function is not None:
            signal = self.norm_function(signal)
        signal = torch.tensor(signal)
        if torch.isnan(signal).any():
            raise ValueError(f"NaN values found in signal from file {file_path}")
        label = torch.tensor(label)
        if self.target_transform:
            label = self.target_transform(label)
        
        
        return signal, label, idx
    
    def compute_global_min_max(self):
        global_min = float('inf')
        global_max = float('-inf')
        for file_path, _ in self.segment_lists['train']:
            try:
                sr, signal = wavfile.read(file_path, mmap=False)
                signal = signal.astype(np.float32)
                file_min = np.min(signal)
                file_max = np.max(signal)
                if file_min < global_min:
                    global_min = file_min
                if file_max > global_max:
                    global_max = file_max
            except Exception as e:
                raise RuntimeError(f"Error reading file {file_path}: {e}")
        return global_min, global_max

    def set_norm_function(self, global_min, global_max):
        self.norm_function = lambda x: (x - global_min) / (global_max - global_min)
        print(f"Normalization function set with global_min: {global_min}, global_max: {global_max}")
class DeepShipDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, pin_memory, 
                 train_split=0.7, val_split=0.1, test_split=0.2, random_seed=42, shuffle=True):
        super().__init__()
        print(f"train_split: {train_split}, val_split: {val_split}, test_split: {test_split}")
        assert train_split + val_split + test_split == 1, "Splits must add up to 1"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.norm_function = None
        self.global_min = None
        self.global_max = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DeepShipSegments(self.data_dir, partition='train', 
                                                  train_split=self.train_split,
                                                  val_split=self.val_split, 
                                                  test_split=self.test_split, 
                                                  random_seed=self.random_seed, 
                                                  shuffle=self.shuffle)
            self.val_dataset = DeepShipSegments(self.data_dir, partition='val', 
                                                train_split=self.train_split, 
                                                val_split=self.val_split, 
                                                test_split=self.test_split, 
                                                random_seed=self.random_seed, 
                                                shuffle=self.shuffle)
            
            self.global_min, self.global_max = self.train_dataset.compute_global_min_max()
            self.norm_function = lambda x: (x - self.global_min) / (self.global_max - self.global_min)
            self.train_dataset.set_norm_function(self.global_min, self.global_max)
            self.val_dataset.set_norm_function(self.global_min, self.global_max)

        if stage == 'test' or stage is None:
            self.test_dataset = DeepShipSegments(self.data_dir, partition='test', 
                                                 train_split=self.train_split, 
                                                 val_split=self.val_split, 
                                                 test_split=self.test_split, 
                                                 random_seed=self.random_seed, 
                                                 shuffle=self.shuffle)
            if self.global_min is None or self.global_max is None:
                raise RuntimeError("Global min and max must be computed during the 'fit' stage before testing.")
            self.test_dataset.set_norm_function(self.global_min, self.global_max)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size['train'], 
                          shuffle=True, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size['val'], 
                          shuffle=False, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size['test'], 
                          shuffle=False, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory)