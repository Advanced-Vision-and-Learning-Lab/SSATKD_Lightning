# Demo.py

This script is designed for training and evaluating models for experiments on datasets.

## Usage

1. Ensure you have Python installed.
2. Clone this repository.
3. Install the required dependencies listed in `requirements.txt`.
4. Navigate to the root directory of the cloned repository.
5. Run the following command to execute the script:
   ```
   python Demo.py
   ```

## Important Parameters

- `--save_results`: Flag to save results of experiments (default: True)
- `--folder`: Location to save models (default: 'Saved_Models')
- `--data_selection`: Dataset selection:
  - 1: FashionMNIST
  - 2: CIFAR10
  - 3: sugarcane_damage_usa
- `--feature_extraction`: Flag for feature extraction:
  - False: Train whole model
  - True: Only update fully connected/encoder parameters (default: False)
- `--use_pretrained`: Flag to use pretrained model from ImageNet or train from scratch (default: True)
- `--num_workers`: Number of workers for dataloader (default: 0)
- `--Parallelize`: Enables parallel functionality (default: True)
- `--train_batch_size`: Input batch size for training (default: 4)
- `--val_batch_size`: Input batch size for validation (default: 32)
- `--test_batch_size`: Input batch size for testing (default: 32)
- `--num_epochs`: Number of epochs to train each model for (default: 10)
- `--resize_size`: Resize the image before center crop (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--model`: Backbone architecture to use (default: 'simple_cnn')

## Dependencies

- `numpy`
- `torch`
- `matplotlib`
- `pytorch_lightning`

## License

[MIT License](LICENSE)

# View_Results.py

This script is designed for generating results from saved models.

## Usage

1. Ensure you have Python installed.
2. Clone this repository.
3. Install the required dependencies listed in `requirements.txt`.
4. Navigate to the root directory of the cloned repository.
5. After training through demo.py, run the following command to execute the script:
   ```
   python View_Results.py
   ```

## Parameters

Use the same parameters as demo.py

**Additional Parameters**
TBD

## Dependencies

- `numpy`
- `matplotlib`
- `pytorch`
- `pytorch_lightning`

## Using Tensorboard with View Results

**General documentation is [here](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)**

Access the tensorboard interface by typing the below command

```
tensorboard --logdir <ModelDirectory>
```

Using tensorboard in this repo, you can see metrics collected as well as the loss curves. Please note, tensorboard aggregates this data using batch not Epochs. You can also export this data to make more appropriate visualizations.

The scalar metrics can be found in both the tensorboard and the aggregated_results.json file.

## License

[MIT License](LICENSE)

# DataModules.py

This script provides PyTorch Lightning DataModules for various datasets to be used for training and evaluation.

## Usage

This script contains DataModules for three different datasets: FashionMNIST, CIFAR10, and sugarcane_damage_usa. To use these DataModules, follow these steps:

1. Import the desired DataModule class from this script into your main script.
2. Initialize an instance of the DataModule class with the appropriate parameters.
3. Call the `prepare_data()` method to download and prepare the dataset.
4. Call the `setup()` method to split the dataset into training, validation, and test sets and apply any necessary transformations.
5. Use the `train_dataloader()`, `val_dataloader()`, and `test_dataloader()` methods to obtain DataLoader objects for training, validation, and testing, respectively.

## Adjustments

You can adjust the behavior of the DataModules by modifying the parameters passed to the constructor of each DataModule class. Additionally, you can modify the transformations applied to the data by editing the `train_transform` and `test_transform` attributes.

## Adding Your Own Dataset

To add your own dataset, follow these general instructions:

1. Ensure your dataset is formatted appropriately and accessible from your local filesystem or a remote location.
2. Create a new class in this script that inherits from `L.LightningDataModule` (where `L` is the alias for PyTorch Lightning) and implements the necessary methods (`prepare_data()`, `setup()`, `train_dataloader()`, `val_dataloader()`, `test_dataloader()`).
3. Define any necessary transformations for your dataset and apply them within the `train_transform` and `test_transform` attributes of your DataModule class.
4. Update the `prepare_data()` method to download and prepare your dataset.
5. Update the `setup()` method to split your dataset into training, validation, and test sets and apply the defined transformations.
6. Import and use your custom DataModule class in your main script following the same steps as described in the "Usage" section.

## Dependencies

- `lightning`
- `torch`
- `albumentations`
- `torchvision`

## License

[MIT License](LICENSE)

# Demo_Parameters.py

This script contains parameters for training and evaluation of models. If you want to add your own dataset to the training pipeline, you'll need to make adjustments to the parameters in this script.

## Adjustments for Adding Your Own Dataset

1. **Dataset Selection (`data_selection`):** Choose a unique integer identifier for your dataset and update the `Dataset_names` dictionary to include the mapping from your identifier to the dataset name.

2. **Dataset Directory (`Data_dirs`):** Specify the directory path where your dataset is located and update the `Data_dirs` dictionary with the dataset name as the key and the directory path as the value.

3. **Class Names (`Class_names`):** Provide the list of class names for your dataset and update the `Class_names` dictionary with the dataset name as the key and the list of class names as the value.

4. **Number of Channels (`channels`):** Specify the number of channels in your dataset (e.g., 1 for grayscale, 3 for RGB) and update the `channels` dictionary with the dataset name as the key and the number of channels as the value.

5. **Number of Classes (`num_classes`):** Define the number of classes in your dataset and update the `num_classes` dictionary with the dataset name as the key and the number of classes as the value.

6. **Number of Runs/Splits (`Splits`):** Determine the number of runs or splits for your dataset and update the `Splits` dictionary with the dataset name as the key and the number of runs/splits as the value.

7. **Data Directory (`data_dir`):** Specify the directory path where your dataset is located based on the selected dataset.

8. **Class Names (`class_names`):** Update the `class_names` variable with the list of class names for your dataset based on the selected dataset.

## Usage

After adjusting the parameters for your dataset, you can use the `Parameters` function to obtain the dictionary of parameters, which can then be used in your main script for training and evaluation.

## Dependencies

This script has no external dependencies.

## Creating Conda Env on HPRC, Loading the GPU

This repo was made with Python 3.11.5. For use on HPRC refer to the following instructions:

1. Navigate to the project directory then load Python 3.11.5 and load Cuda using the following commands (always do that when using this framework)

```
module load GCCcore/13.2.0 Python/3.11.5
ml CUDA
```

2. Create a virtual environment if one isnt in use.

```
python -m venv venv
```

3. Activate the venv (you will need to always activate the venv)

```
source venv/bin/activate
```

4. Install the Requirements

```
pip install -r requirements.txt
```

5. Install PyTorch with GPU support

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

## License

[MIT License](LICENSE)
