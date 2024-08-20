import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

#log_dir = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/tb_logs/model_logs/run_1'
log_dir = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/loss_test_SGD/Adagrad/distillation/Fine_Tuning/DeepShip/TDNN_CNN_14/Run_1/tb_logs/model_logs/run_1'

def list_all_tags(logdir):
    tags_found = set()
    for root, dirs, files in os.walk(logdir):
        for file_name in files:
            if file_name.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file_name)
                print(f"Reading file: {event_file}")
                for event in tf.compat.v1.train.summary_iterator(event_file):
                    for value in event.summary.value:
                        tags_found.add(value.tag)
    return tags_found

# List and print all tags
log_dir = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/AdamW0.001CNN14_3runs/Adagrad/teacher/Pretrained/Fine_Tuning/DeepShip/CNN_14/Run_2/tb_logs/model_logs/version_0'
tags_found = list_all_tags(log_dir)
print("\nTags found in the log files:")
for tag in tags_found:
    print(tag)

# Function to extract scalar data from TensorBoard logs with debugging
def extract_scalars(logdir, tags):
    scalars = {tag: [] for tag in tags}
    epochs = {tag: [] for tag in tags}
    current_epoch = None
    
    for root, dirs, files in os.walk(logdir):
        for file_name in files:
            if file_name.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file_name)
                print(f"Reading file: {event_file}")
                for event in tf.compat.v1.train.summary_iterator(event_file):
                    for value in event.summary.value:
                        if value.tag == 'epoch':
                            current_epoch = int(value.simple_value)
                            print(f"Epoch tag found: {current_epoch} at step {event.step}")
                        if value.tag in tags:
                            scalars[value.tag].append(value.simple_value)
                            epochs[value.tag].append(current_epoch)
                            print(f"Found tag: {value.tag}, Epoch: {current_epoch}, Value: {value.simple_value}, Step: {event.step}")
    
    # Fill missing epoch values
    for tag in tags:
        df = pd.DataFrame({'epoch': epochs[tag], 'value': scalars[tag]})
        df['epoch'] = df['epoch'].fillna(method='ffill').astype(int)
        epochs[tag] = df['epoch'].tolist()
        scalars[tag] = df['value'].tolist()

    return scalars, epochs

# List of tags to extract
tags = [
    'epoch',
    'struct_loss',
    'stats_loss',
    'train_accuracy',
    'classification_loss',
    'val_accuracy',
    'val_loss',
    'distillation_loss',
    'train_loss',
    'val_classification_loss',
    'val_distillation_loss',
    'val_struct_loss',
    'val_stats_loss'
]

# Extract data
scalars, epochs = extract_scalars(log_dir, tags)
print("Extraction complete.")

# Print the extracted data for verification
print("\nExtracted Data:")
for tag in tags:
    print(f"\nTag: {tag}")
    print(f"Epochs: {epochs[tag]}")
    print(f"Values: {scalars[tag]}")

# Function to plot scalar data with debugging
def plot_scalar(tag, x, values):
    if len(x) == 0 or len(values) == 0:
        print(f"No data to plot for tag: {tag}")
        return
    plt.plot(x, values, label=tag)

# Plot train and validation losses separately
def plot_train_val_losses_separately(train_epochs, train_values, val_epochs, val_values, loss_type):
    plt.figure(figsize=(10, 6))
    plot_scalar(f'train_{loss_type}', train_epochs, train_values)
    plot_scalar(f'val_{loss_type}', val_epochs, val_values)
    plt.xlabel('Epochs')
    plt.ylabel(loss_type.capitalize() + ' Loss')
    plt.title(f'Training and Validation {loss_type.capitalize()} Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{loss_type}_loss_plot.png')
    plt.show()

# Plot train and validation accuracies together
def plot_train_val_accuracies(train_epochs, train_values, val_epochs, val_values):
    plt.figure(figsize=(10, 6))
    plot_scalar('train_accuracy', train_epochs, train_values)
    plot_scalar('val_accuracy', val_epochs, val_values)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.show()

# Plot train and val accuracies together
print("Plotting Training and Validation Accuracies:")
plot_train_val_accuracies(epochs['train_accuracy'], scalars['train_accuracy'], epochs['val_accuracy'], scalars['val_accuracy'])

# Plot train and val losses separately
print("Plotting Training and Validation Losses:")
plot_train_val_losses_separately(epochs['train_loss'], scalars['train_loss'], epochs['val_loss'], scalars['val_loss'], 'loss')
plot_train_val_losses_separately(epochs['classification_loss'], scalars['classification_loss'], epochs['val_classification_loss'], scalars['val_classification_loss'], 'classification')
plot_train_val_losses_separately(epochs['distillation_loss'], scalars['distillation_loss'], epochs['val_distillation_loss'], scalars['val_distillation_loss'], 'distillation')
plot_train_val_losses_separately(epochs['struct_loss'], scalars['struct_loss'], epochs['val_struct_loss'], scalars['val_struct_loss'], 'struct')
plot_train_val_losses_separately(epochs['stats_loss'], scalars['stats_loss'], epochs['val_stats_loss'], scalars['val_stats_loss'], 'stats')


# def save_learning_curves(log_dir, output_path):
#     # Get all event files in the log directory
#     event_paths = []
#     for root, dirs, files in os.walk(log_dir):
#         for file in files:
#             if file.startswith('events.out.tfevents.'):
#                 event_paths.append(os.path.join(root, file))

#     # List all available scalars
#     available_scalars = list_available_scalars(event_paths)
#     print("Available Scalars:", available_scalars)

#     # Use correct scalar names after checking available scalars
#     train_loss = extract_scalar_from_events(event_paths, 'loss_epoch')
#     val_loss = extract_scalar_from_events(event_paths, 'val_loss')

#     # Plot learning curves if both scalars are available
#     if train_loss and val_loss:
#         plt.figure(figsize=(8, 6))
#         plt.plot(train_loss, label='Training Loss', color='blue', lw=2)
#         plt.plot(val_loss, label='Validation Loss', color='orange', lw=2)
#         plt.xlabel('Epochs', fontsize=15)
#         plt.ylabel('Loss', fontsize=15)
#         plt.title('Learning Curves', fontsize=18)
#         plt.legend(loc="best", fontsize=12)
#         plt.grid(True)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)

#         # Save the figure
#         plt.savefig(output_path, dpi=300)
#         plt.close()
#     else:
#         print("Required scalars ('loss_epoch', 'val_loss') not found in event files.")

# def list_available_scalars(event_paths):
#     available_scalars = set()
#     for event_path in event_paths:
#         event_acc = EventAccumulator(event_path)
#         event_acc.Reload()
#         available_scalars.update(event_acc.Tags()['scalars'])
#     return available_scalars


# def extract_scalar_from_events(event_paths, scalar_name):
#     scalar_values = []
#     for event_path in event_paths:
#         event_acc = EventAccumulator(event_path)
#         event_acc.Reload()
#         if scalar_name in event_acc.Tags()['scalars']:
#             scalar_events = event_acc.Scalars(scalar_name)
#             values = [event.value for event in scalar_events]
#             scalar_values.extend(values)
#     return scalar_values
# output = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/HLTDNN_300_event'
# save_learning_curves(log_dir,output)


# # Function to list all available tags in TensorBoard logs
# def list_all_tags(logdir):
#     tags_found = set()
#     for file_name in os.listdir(logdir):
#         if file_name.startswith('events.out.tfevents'):
#             event_file = os.path.join(logdir, file_name)
#             print(f"Reading file: {event_file}")
#             for event in tf.compat.v1.train.summary_iterator(event_file):
#                 for value in event.summary.value:
#                     tags_found.add(value.tag)
#     return tags_found

# # List and print all tags
# tags_found = list_all_tags(log_dir)
# print("\nTags found in the log files:")
# for tag in tags_found:
#     print(tag)
# # pdb.set_trace()
# # Function to extract scalar data from TensorBoard logs with debugging
# def extract_scalars(logdir, tags):
#     scalars = {tag: [] for tag in tags}
#     steps = {tag: [] for tag in tags}
#     for file_name in os.listdir(logdir):
#         if file_name.startswith('events.out.tfevents'):
#             event_file = os.path.join(logdir, file_name)
#             print(f"Reading file: {event_file}")
#             for event in tf.compat.v1.train.summary_iterator(event_file):
#                 for value in event.summary.value:
#                     if value.tag in tags:
#                         scalars[value.tag].append(value.simple_value)
#                         steps[value.tag].append(event.step)
#                         print(f"Found tag: {value.tag}, Step: {event.step}, Value: {value.simple_value}")
#     return steps, scalars

# # List of tags to extract, including test_accuracy
# tags = [
#     'epoch',
#     'struct_loss',
#     'stats_loss',
#     'train_accuracy',
#     'classification_loss',
#     'val_accuracy',
#     'val_loss',
#     'distillation_loss',
#     'train_loss',
#     'val_classification_loss',
#     'val_distillation_loss',
#     'val_struct_loss',
#     'val_stats_loss',
#     'test_acc'
#     'test_loss'
# ]

# # Extract data
# steps, scalars = extract_scalars(log_dir, tags)
# print("Extraction complete.")

# # Function to plot scalar data with debugging
# def plot_scalar(tag, steps, values):
#     if len(steps) == 0 or len(values) == 0:
#         print(f"No data to plot for tag: {tag}")
#         return
#     plt.plot(steps, values, label=tag)

# # Plot train and validation losses separately
# def plot_train_val_losses_separately(train_steps, train_values, val_steps, val_values, test_steps, test_values, loss_type):
#     plt.figure(figsize=(10, 6))
#     plot_scalar(f'train_{loss_type}', train_steps, train_values)
#     plot_scalar(f'val_{loss_type}', val_steps, val_values)
#     plot_scalar(f'test_{loss_type}', test_steps, val_values)
#     plt.xlabel('Steps')
#     plt.ylabel(loss_type.capitalize() + ' Loss')
#     plt.title(f'Training and Validation {loss_type.capitalize()} Loss over Steps')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f'{loss_type}_loss_plot.png')
#     plt.show()

# # Plot train and validation accuracies together
# def plot_train_val_accuracies(train_steps, train_values, val_steps, val_values, test_steps, test_values):
#     plt.figure(figsize=(10, 6))
#     plot_scalar('train_accuracy', train_steps, train_values)
#     plot_scalar('val_accuracy', val_steps, val_values)
#     plot_scalar('test_accuracy', test_steps, test_values)
#     plt.xlabel('Steps')
#     plt.ylabel('Accuracy')
#     plt.title('Training, Validation, and Test Accuracy over Steps')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('accuracy_plot.png')
#     plt.show()

# # Plot train, val, and test accuracies together
# print("Plotting Training, Validation, and Test Accuracies:")
# plot_train_val_accuracies(
#     steps['train_accuracy'], scalars['train_accuracy'],
#     steps['val_accuracy'], scalars['val_accuracy'],
#     steps['test_acc'], scalars['test_acc']
# )

# # Plot train and val losses separately
# print("Plotting Training and Validation Losses:")
# plot_train_val_losses_separately(steps['train_loss'], scalars['train_loss'], steps['val_loss'], scalars['val_loss'], 'loss')
# plot_train_val_losses_separately(steps['classification_loss'], scalars['classification_loss'], steps['val_classification_loss'], scalars['val_classification_loss'], 'classification')
# plot_train_val_losses_separately(steps['distillation_loss'], scalars['distillation_loss'], steps['val_distillation_loss'], scalars['val_distillation_loss'], 'distillation')
# plot_train_val_losses_separately(steps['struct_loss'], scalars['struct_loss'], steps['val_struct_loss'], scalars['val_struct_loss'], 'struct')
# plot_train_val_losses_separately(steps['stats_loss'], scalars['stats_loss'], steps['val_stats_loss'], scalars['val_stats_loss'], 'stats')

# # Print final accuracy values
# if 'train_accuracy' in scalars and scalars['train_accuracy']:
#     final_train_accuracy = np.mean(scalars['train_accuracy'])
#     print(f"Final Train Accuracy: {final_train_accuracy:.4f}")

# if 'val_accuracy' in scalars and scalars['val_accuracy']:
#     final_val_accuracy = np.mean(scalars['val_accuracy'])
#     print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

# if 'test_accuracy' in scalars and scalars['test_acc']:
#     final_test_accuracy = np.mean(scalars['test_acc'])
#     print(f"Final Test Accuracy: {final_test_accuracy:.4f}")

# # # Save final accuracies to a text file
# # with open('final_accuracies.txt', 'w') as f:
# #     f.write(f"Final Train Accuracy: {final_train_accuracy:.4f}\n")
# #     f.write(f"Final Validation Accuracy: {final_val_accuracy:.4f}\n")
# #     f.write(f"Final Test Accuracy: {final_test_accuracy:.4f}\n")