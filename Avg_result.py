import os
import tensorflow as tf
import numpy as np

def extract_accuracies(events_file):
    accuracies = {'val_accuracy': [], 'test_accuracy': []}
    for event in tf.compat.v1.train.summary_iterator(events_file):
        for value in event.summary.value:
            if value.tag == 'val_accuracy':
                accuracies['val_accuracy'].append(value.simple_value)
            elif value.tag == 'test_accuracy':
                accuracies['test_accuracy'].append(value.simple_value)
    return accuracies

def calculate_statistics(accuracies_list):
    val_accuracies = [np.mean(acc['val_accuracy']) for acc in accuracies_list]
    test_accuracies = [np.mean(acc['test_accuracy']) for acc in accuracies_list]

    avg_val_accuracy = np.mean(val_accuracies)
    std_val_accuracy = np.std(val_accuracies)

    avg_test_accuracy = np.mean(test_accuracies)
    std_test_accuracy = np.std(test_accuracies)

    return {
        'avg_val_accuracy': avg_val_accuracy,
        'std_val_accuracy': std_val_accuracy,
        'avg_test_accuracy': avg_test_accuracy,
        'std_test_accuracy': std_test_accuracy,
    }

def find_event_file(run_path):
    lightning_logs_path = os.path.join(run_path, 'lightning_logs')
    print(f"Checking path: {lightning_logs_path}")
    
    if not os.path.exists(lightning_logs_path):
        print(f"Directory does not exist: {lightning_logs_path}")
        print(f"Available directories in {run_path}: {os.listdir(run_path)}")
        return None
    
    training_path = os.path.join(lightning_logs_path, 'Training')
    if not os.path.exists(training_path):
        print(f"Directory does not exist: {training_path}")
        print(f"Available directories in {lightning_logs_path}: {os.listdir(lightning_logs_path)}")
        return None
    
    for file in os.listdir(training_path):
        if file.startswith('events.out.tfevents'):
            return os.path.join(training_path, file)
    
    print(f"No event files found in {training_path}")
    return None

base_path = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/MobileNetV1_32khz/Adagrad/teacher/Pretrained/Fine_Tuning/DeepShip/MobileNetV1'
runs = ['Run1', 'Run2', 'Run3']

all_accuracies = []
for run in runs:
    run_path = os.path.join(base_path, run)
    print(f"Constructed path for {run}: {run_path}")  # Print the full constructed path
    events_file = find_event_file(run_path)
    if events_file:
        accuracies = extract_accuracies(events_file)
        all_accuracies.append(accuracies)
    else:
        print(f"No event file found for {run}")

if all_accuracies:
    statistics = calculate_statistics(all_accuracies)

    print(f"Average Validation Accuracy: {statistics['avg_val_accuracy']}")
    print(f"Standard Deviation of Validation Accuracy: {statistics['std_val_accuracy']}")
    print(f"Average Test Accuracy: {statistics['avg_test_accuracy']}")
    print(f"Standard Deviation of Test Accuracy: {statistics['std_test_accuracy']}")
else:
    print("No accuracies found. Please check the directories and event files.")
