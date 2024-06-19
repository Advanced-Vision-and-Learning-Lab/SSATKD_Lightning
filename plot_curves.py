import tensorflow as tf
import matplotlib.pyplot as plt
import os


# current_directory = os.getcwd()
# print(f"Current Directory: {current_directory}")

# # List the contents of the current directory
# directory_contents = os.listdir(current_directory)
# print("Directory Contents:")
# for item in directory_contents:
#     print(f" - {item}")
    
#     # Define each segment of the path
# base_path = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning'
# segments = [
#     'Saved_Models',
#     'KD_Test',
#     'Adagrad',
#     'distillation',
#     'Fine_Tuning',
#     'DeepShip',
#     'TDNN_CNN_14',
#     'Run_1',
#     'lightning_logs',
#     'Training'
# ]


# Define the path to the log directory
log_dir = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/loss/distillation/Feature_Extraction/DeepShip/TDNN_CNN_14/Run_1/lightning_logs/Training'

# Function to extract scalar data from TensorBoard logs with debugging
def extract_scalars(logdir, tags):
    scalars = {tag: [] for tag in tags}
    steps = {tag: [] for tag in tags}
    for file_name in os.listdir(logdir):
        if file_name.startswith('events.out.tfevents'):
            event_file = os.path.join(logdir, file_name)
            print(f"Reading file: {event_file}")
            for event in tf.compat.v1.train.summary_iterator(event_file):
                for value in event.summary.value:
                    if value.tag in tags:
                        scalars[value.tag].append(value.simple_value)
                        steps[value.tag].append(event.step)
                        print(f"Found tag: {value.tag}, Step: {event.step}, Value: {value.simple_value}")
    return steps, scalars

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
    'train_loss'
]

# Extract data
steps, scalars = extract_scalars(log_dir, tags)
print("Extraction complete.")

# Debugging: Print the collected data
for tag in tags:
    print(f"{tag} - Steps: {steps[tag]}")
    print(f"{tag} - Scalars: {scalars[tag]}")
    
    # Function to plot scalar data with debugging
def plot_scalar(tag, steps, values):
    if len(steps) == 0 or len(values) == 0:
        print(f"No data to plot for tag: {tag}")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label=tag)
    plt.xlabel('Steps')
    plt.ylabel(tag)
    plt.title(f'{tag} over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot each tag separately
for tag in tags:
    print(f"Plotting data for tag: {tag}")
    plot_scalar(tag, steps[tag], scalars[tag])
# # Function to extract and print all tags from TensorBoard logs
# def extract_all_tags(logdir):
#     tags = set()
#     for file_name in os.listdir(logdir):
#         if file_name.startswith('events.out.tfevents'):
#             event_file = os.path.join(logdir, file_name)
#             for event in tf.compat.v1.train.summary_iterator(event_file):
#                 for value in event.summary.value:
#                     tags.add(value.tag)
#     return tags

# # Extract all tags
# all_tags = extract_all_tags(log_dir)

# print("Tags found in TensorBoard logs:")
# for tag in all_tags:
#     print(tag)
    
# # Function to extract scalar data from TensorBoard logs
# def extract_scalar_events(logdir, tag):
#     scalar_events = []
#     for file_name in sorted(os.listdir(logdir)):
#         if file_name.startswith('events.out.tfevents'):
#             event_file = os.path.join(logdir, file_name)
#             for event in tf.compat.v1.train.summary_iterator(event_file):
#                 for value in event.summary.value:
#                     if value.tag == tag:
#                         scalar_events.append((event.step, value.simple_value))
#     return scalar_events

# # Extract data for all relevant metrics
# train_loss_events = extract_scalar_events(log_dir, 'train_loss')
# val_loss_events = extract_scalar_events(log_dir, 'val_loss')
# train_accuracy_events = extract_scalar_events(log_dir, 'train_accuracy')
# val_accuracy_events = extract_scalar_events(log_dir, 'val_accuracy')
# epoch_events = extract_scalar_events(log_dir, 'epoch')

# # Separate steps and values for each metric
# def separate_steps_values(events):
#     if events:
#         steps, values = zip(*events)
#     else:
#         steps, values = [], []
#     return steps, values

# train_steps, train_loss = separate_steps_values(train_loss_events)
# val_steps, val_loss = separate_steps_values(val_loss_events)
# train_acc_steps, train_accuracy = separate_steps_values(train_accuracy_events)
# val_acc_steps, val_accuracy = separate_steps_values(val_accuracy_events)
# epoch_steps, epochs = separate_steps_values(epoch_events)

# # Plotting the loss and accuracy curves
# plt.figure(figsize=(12, 8))

# # Training and validation loss
# plt.subplot(2, 1, 1)
# plt.plot(train_steps, train_loss, label='Training Loss')
# plt.plot(val_steps, val_loss, label='Validation Loss')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.grid(True)

# # Training and validation accuracy
# plt.subplot(2, 1, 2)
# plt.plot(train_acc_steps, train_accuracy, label='Training Accuracy')
# plt.plot(val_acc_steps, val_accuracy, label='Validation Accuracy')
# plt.xlabel('Steps')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Print out the final values of each metric
# if train_loss:
#     print(f"Final Training Loss: {train_loss[-1]}")
# if val_loss:
#     print(f"Final Validation Loss: {val_loss[-1]}")
# if train_accuracy:
#     print(f"Final Training Accuracy: {train_accuracy[-1]}")
# if val_accuracy:
#     print(f"Final Validation Accuracy: {val_accuracy[-1]}")
# if epochs:
#     print(f"Total Epochs: {epochs[-1]}")
    
