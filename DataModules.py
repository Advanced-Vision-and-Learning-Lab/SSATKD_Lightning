"""
Created on Thursday April 25 22:32:00 2024
Wrap the data into a PyTorch Lightning DataModule for training and evaluation
@author: salimalkharsa, jpeeples
"""
import lightning as L
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
import torch
import agml

class FashionMNIST_DataModule(L.LightningDataModule):
    def __init__(self, resize_size, input_size, data_dir: str = "path/to/dir", 
                 batch_size: dict = {'train': 16,'val': 64, 'test': 64}, num_workers: int = 0):
        super().__init__()
        # Pass these through the Network Parameters when calling the DataModule
        self.resize_size = resize_size
        self.input_size = input_size
        
        # Other parameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Set a temporary mean of 0.5 and std of 0.5
        # TODO: Calculate the mean by function that takes in the dataset
        mean = [0.5]
        std = [0.5]
        # Define the transform
        self.train_transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        self.test_transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def prepare_data(self):
        # download the dataset
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full_data = datasets.FashionMNIST(self.data_dir, train=True, transform=self.train_transform)
            # For debug purpose trim the dataset to 100 samples
            #full_data = torch.utils.data.Subset(full_data, torch.arange(1000))
            self.fashionmnist_train, self.fashionmnist_val = random_split(
                full_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.fashionmnist_test = datasets.FashionMNIST(self.data_dir, 
                                                           train=False, 
                                                           transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.fashionmnist_train, batch_size=self.batch_size['train'], num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.fashionmnist_val, batch_size=self.batch_size['val'], num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.fashionmnist_test, batch_size=self.batch_size['test'], num_workers=self.num_workers)
    
class CIFAR10_DataModule(L.LightningDataModule):
    def __init__(self, resize_size, input_size, data_dir: str = "path/to/dir", 
                 batch_size: dict = {'train': 16,'val': 64, 'test': 64}, num_workers: int = 0):
        super().__init__()
        # Pass these through the Network Parameters when calling the DataModule
        self.resize_size = resize_size
        self.input_size = input_size
        # Other parameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Set a hard coded mean and std for CIFAR10
        # TODO: Calculate the mean by function that takes in the dataset
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Define the transform
        self.train_transform  = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        self.test_transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
    def prepare_data(self):
        # download the dataset
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full_data = datasets.CIFAR10(self.data_dir, train=True, transform=self.train_transform)
            # For debug purpose trim the dataset to 100 samples
            #full_data = torch.utils.data.Subset(full_data, torch.arange(100))

            self.cifar10_train, self.cifar10_val = random_split(
                full_data, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar10_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.test_transform)
            # For debug purpose trim the dataset to 100 samples
            self.cifar10_test = torch.utils.data.Subset(self.cifar10_test, torch.arange(100))

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size['train'], num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size['val'], num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size['test'], num_workers=self.num_workers)

class sugarcane_damage_usa_DataModule(L.LightningDataModule):
    def __init__(self, resize_size, input_size, data_dir: str = "path/to/dir", 
                 batch_size: dict = {'train': 16,'val': 64, 'test': 64}, 
                 num_workers: int = 0):
        super().__init__()
        # Pass these through the Network Parameters when calling the DataModule
        self.resize_size = resize_size
        self.input_size = input_size
        # Other parameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Set a hard coded mean and std
        mean = [0.244, 0.247, 0.262]
        std = [0.224, 0.217, 0.192]

        # Define Albumentations transformations
        self.train_transform = A.Compose([
            A.Resize(self.resize_size, self.resize_size),
            A.RandomResizedCrop(self.input_size, self.input_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        self.test_transform = A.Compose([
            A.Resize(self.resize_size, self.resize_size),
            A.CenterCrop(self.input_size, self.input_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def prepare_data(self):
        # see if the dataset is already downloaded
        agml.data.AgMLDataLoader('sugarcane_damage_usa', dataset_path=self.data_dir)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        full_data = agml.data.AgMLDataLoader('sugarcane_damage_usa', dataset_path=self.data_dir)
        # Split the data into train/val/test sets.
        full_data.split(train = 0.7, val = 0.1, test = 0.2)
        if stage == "fit":
            # Do the transformations
            self.sugarcane_damage_usa_train = full_data.train_data
            self.sugarcane_damage_usa_train.transform(self.train_transform)
            self.sugarcane_damage_usa_train.as_torch_dataset()

            self.sugarcane_damage_usa_val = full_data.val_data
            self.sugarcane_damage_usa_val.transform(self.test_transform)
            self.sugarcane_damage_usa_val.as_torch_dataset()
        if stage == "test":
            self.sugarcane_damage_usa_test = full_data.test_data
            self.sugarcane_damage_usa_test.transform(self.test_transform)
            self.sugarcane_damage_usa_test.as_torch_dataset()
            

    def train_dataloader(self):
        return DataLoader(self.sugarcane_damage_usa_train, batch_size=self.batch_size['train'], num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.sugarcane_damage_usa_val, batch_size=self.batch_size['val'], num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.sugarcane_damage_usa_test, batch_size=self.batch_size['test'], num_workers=self.num_workers)
