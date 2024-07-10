#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:01:09 2024

@author: jarin.ritu
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Define the path to the log directory
log_dir = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/student_split/Adagrad/student/Fine_Tuning/DeepShip/TDNN/Run_1/lightning_logs/Training'


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
    'train_accuracy',
    'val_accuracy',
    'val_loss',
    'train_loss'
]

# Extract data
steps, scalars = extract_scalars(log_dir, tags)
print("Extraction complete.")

# Function to plot scalar data with debugging
def plot_scalar(tag, steps, values):
    if len(steps) == 0 or len(values) == 0:
        print(f"No data to plot for tag: {tag}")
        return
    plt.plot(steps, values, label=tag)

# Plot train and validation losses together
def plot_train_val_losses(train_steps, train_values, val_steps, val_values):
    plt.figure(figsize=(10, 6))
    plot_scalar('train_loss', train_steps, train_values)
    plot_scalar('val_loss', val_steps, val_values)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot train and validation accuracies together
def plot_train_val_accuracies(train_steps, train_values, val_steps, val_values):
    plt.figure(figsize=(10, 6))
    plot_scalar('train_accuracy', train_steps, train_values)
    plot_scalar('val_accuracy', val_steps, val_values)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot train and val losses together
print("Plotting Training and Validation Losses:")
plot_train_val_losses(steps['train_loss'], scalars['train_loss'], steps['val_loss'], scalars['val_loss'])

# Plot train and val accuracies together
print("Plotting Training and Validation Accuracies:")
plot_train_val_accuracies(steps['train_accuracy'], scalars['train_accuracy'], steps['val_accuracy'], scalars['val_accuracy'])

# Print final accuracy values
if 'train_accuracy' in scalars and scalars['train_accuracy']:
    final_train_accuracy = scalars['train_accuracy'][-1]
    print(f"Final Train Accuracy: {final_train_accuracy:.4f}")

if 'val_accuracy' in scalars and scalars['val_accuracy']:
    final_val_accuracy = scalars['val_accuracy'][-1]
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")