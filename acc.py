#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:24:42 2024

@author: jarin.ritu
"""

import torch
import numpy as np

# Function to load accuracy from a checkpoint file
def load_accuracy_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    accuracy = checkpoint.get('accuracy', None)  # Assuming 'accuracy' key stores the accuracy in the checkpoint
    return accuracy

# Example model names and number of runs
student_model = "resnet"
teacher_model = "vgg"
numRuns = 3
accuracies = []

for run_number in range(numRuns):
    checkpoint_path = f"{student_model}_{teacher_model}/Run_{run_number}/tb_logs/model_logs/version_0/checkpoints/best_model.ckpt"
    
    try:
        accuracy = load_accuracy_from_checkpoint(checkpoint_path)
        if accuracy is not None:
            accuracies.append(accuracy)
        else:
            print(f"Accuracy not found in checkpoint for run {run_number}")
    except FileNotFoundError:
        print(f"Checkpoint file not found for run {run_number}")

if accuracies:
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"Average accuracy: {avg_accuracy}")
    print(f"Standard deviation of accuracy: {std_accuracy}")
else:
    print("No accuracies found to calculate statistics")
