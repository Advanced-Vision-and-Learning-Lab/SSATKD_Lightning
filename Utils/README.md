# Aggregate Results Utility

This utility script, `aggregate_results.py`, is designed to aggregate results from saved models, particularly from TensorBoard logs. It provides functions to traverse through a directory structure containing multiple runs of experiments, read event files from each run directory, and aggregate scalar data such as training, validation, and test metrics.

## Functions

### `aggregate_tensorboard_logs(root_dir)`

This function aggregates tensorboard logs from the specified root directory.

- **Parameters**:

  - `root_dir`: The root directory containing the saved model runs.

- **Returns**:
  - `final_aggregated_results`: A dictionary containing aggregated results with metrics as keys and phase-wise statistics (mean and standard deviation) as values.

# Custom Models

The `custom_models.py` script provides the flexibility to define custom PyTorch neural network architectures for use in your machine learning experiments. In this example, a simple baseline CNN (Convolutional Neural Network) named `Simple_CNN` is provided.

## Usage

You can use this script to define your own custom neural network architectures tailored to your specific tasks and datasets.

### `Simple_CNN`

The `Simple_CNN` class defines a basic CNN architecture with multiple convolutional layers followed by max-pooling and an adaptive average pooling layer. It is designed to work with grayscale or RGB images, as indicated by the `in_channels` parameter. The number of output classes is specified by the `num_classes` parameter.

#### Constructor Parameters

- `in_channels`: Number of input channels (1 for grayscale, 3 for RGB).
- `num_classes`: Number of output classes.

#### Architecture

1. Convolutional layers (`conv1` to `conv5`) with ReLU activation.
2. Max-pooling layer (`maxpool1`) after each convolutional layer.
3. Adaptive average pooling layer (`avgpool`) to ensure a fixed output size.
4. Fully connected layer (`fc`) for classification.

#### Forward Propagation

The `forward` method defines the forward pass of the network, where input data `x` is passed through the convolutional layers, followed by pooling, average pooling, flattening, and finally the fully connected layer.

## Example

You can build your own model by following the general template provided in Simple_CNN

## Generate_TSNE_visual.py

**TBD**

# Lightning Wrapper

The `Lightning_Wrapper` class in `lightning_wrapper.py` is designed to wrap PyTorch models into a Lightning Module for training and evaluation. It simplifies the process of training and evaluating models by handling the training functions and providing convenient logging functionalities.

## Usage

You can use the `Lightning_Wrapper` class to wrap your own PyTorch models and customize the training process according to your specific requirements.

### Constructor Parameters

- `model`: The PyTorch model to be wrapped.
- `num_classes`: Number of classes in the classification task.
- `optimizer`: The optimizer used for training (default is Adam).
- `learning_rate`: Learning rate for the optimizer (default is 1e-3).
- `scheduler`: Learning rate scheduler (optional).
- `criterion`: Loss function (default is CrossEntropyLoss).
- `log_dir`: Directory to save logs and outputs.
- `label_names`: Names of the classes (optional, used for confusion matrix).
- `stage`: Stage of the training process ('train', 'val', or 'test').

### Methods

- `forward`: Forward pass through the model.
- `configure_optimizers`: Configuration of the optimizer.
- `training_step`: Training step for one batch.
- `validation_step`: Validation step for one batch.
- `test_step`: Test step for one batch.
- `log_metrics`: Logging of evaluation metrics (accuracy, F1 score, precision, recall) and confusion matrix.

## Customization

You may need to customize the following aspects of the `Lightning_Wrapper` class to suit your specific use case:

1. **Model Architecture**: You can replace the provided `model` with your own custom PyTorch model architecture.
2. **Loss Function**: Modify the `criterion` parameter to use a different loss function if needed.
3. **Metrics**: Customize the evaluation metrics used by modifying or adding metrics in the `log_metrics` method.
4. **Logging Directory**: Specify the `log_dir` parameter to save logs and outputs in a directory of your choice.

## Example

Here's an example of how to use the `Lightning_Wrapper` class to train and evaluate a PyTorch model:

```python
import torch
from custom_models import Simple_CNN
from lightning_wrapper import Lightning_Wrapper

# Define the model architecture
model = Simple_CNN(in_channels=3, num_classes=10)

# Wrap the model with Lightning_Wrapper
lightning_model = Lightning_Wrapper(model=model, num_classes=10, log_dir='./logs')

# Train the model
trainer = L.Trainer(max_epochs=10, gpus=1)
trainer.fit(lightning_model)

# Evaluate the model
trainer.test(lightning_model)
```

# Network Functions

The `Network_functions.py` script provides functions to load and initialize models, either using common architectures from the Hugging Face Timm library or custom models.

## Usage

You can use the `initialize_model` function to load and initialize models based on their names. This function supports both standard architectures found in the Timm library and custom model implementations.

### `initialize_model` Function

```python
def initialize_model(model_name, use_pretrained=False,
                     num_classes=1, feature_extract=False,
                     channels=3,
                     R=1, measure='norm', p=2.0, similarity=True):
    """
    Initialize a model based on its name.

    Args:
        model_name (str): Name of the model architecture.
        use_pretrained (bool): Whether to use pretrained weights (default is False).
        num_classes (int): Number of classes in the classification task (default is 1).
        feature_extract (bool): Whether to extract features (default is False).
        channels (int): Number of input channels (default is 3).
        R (int): Parameter for custom model (default is 1).
        measure (str): Measure for custom model (default is 'norm').
        p (float): Parameter for custom model (default is 2.0).
        similarity (bool): Whether to compute similarity in custom model (default is True).

    Returns:
        model_ft: Initialized model.
        input_size: Size of the input for the initialized model.
    """
```

# Save Results

The `Save_Results.py` script contains helper functions to save and aggregate results during model training and evaluation.

## Usage

You can use the functions provided in this script to save and aggregate various metrics and results obtained during training and evaluation of models.

### Aggregating Tensorboard Logs

The `aggregate_tensorboard_logs` function aggregates scalar data from Tensorboard logs saved during training. It traverses through the directory structure containing the logs and extracts scalar data for metrics like loss, accuracy, etc.

### Parameters

- `root_dir`: Root directory containing the Tensorboard logs.

### Example

```python
from Save_Results import aggregate_tensorboard_logs

# Provide the root directory containing the Tensorboard logs
root_dir = 'logs/'

# Aggregate Tensorboard logs
results = aggregate_tensorboard_logs(root_dir)

# Print or use the aggregated results as needed
print(results)
```
