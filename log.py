import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_metrics(log_dir, steps_per_epoch):
    event_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if 'events.out.tfevents' in file:
                event_files.append(os.path.join(root, file))

    if not event_files:
        print(f"No event files found in {log_dir}.")
        return {}

    metrics = defaultdict(list)
    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        tags = event_acc.Tags()
        if 'scalars' not in tags:
            continue

        scalar_tags = tags['scalars']
        for tag in scalar_tags:
            events = event_acc.Scalars(tag)
            values = [(e.step // steps_per_epoch, e.value) for e in events]
            metrics[tag].extend(values)

    return metrics

def aggregate_metrics(runs_dirs, steps_per_epoch):
    all_metrics = defaultdict(list)
    for run_dir in runs_dirs:
        print(f"Checking directory: {run_dir}")
        metrics = extract_metrics(run_dir, steps_per_epoch)
        if metrics:
            for key, values in metrics.items():
                all_metrics[key].append(values)
    return all_metrics

def compute_stats(all_metrics):
    stats = {}
    for key, values_list in all_metrics.items():
        # Flatten all values for each tag
        all_values = [value for values in values_list for _, value in values]
        if all_values:
            mean = np.mean(all_values)
            std = np.std(all_values)
            stats[key] = {'mean': mean, 'std': std}
    return stats

def plot_metrics(all_metrics, metric_name):
    plt.figure(figsize=(10, 5))
    for run_idx, values in enumerate(all_metrics.get(metric_name, [])):
        steps = [step for step, _ in values]
        values = [value for _, value in values]
        plt.plot(steps, values, label=f'Run {run_idx + 1}')

    plt.xlabel('Steps')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()
    plt.title(f'{metric_name.replace("_", " ").title()} Across Runs')
    plt.show()

# Example usage:
runs_dirs = [
    '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/AdamW_KD/Adagrad/distillation/Fine_Tuning/DeepShip/TDNN_CNN_14/Run_1/tb_logs/model_logs/run_1'
  ]

steps_per_epoch = 37  # Steps per epoch based on your earlier calculation
all_metrics = aggregate_metrics(runs_dirs, steps_per_epoch)
stats = compute_stats(all_metrics)

# Print detailed metrics for each run
for metric_name, runs_values in all_metrics.items():
    print(f"\nMetric: {metric_name}")
    for run_idx, values in enumerate(runs_values):
        print(f" Run {run_idx + 1}:")
        for step, value in values:
            print(f"  Step {step}: {value}")

# Print aggregated statistics
print("\nAggregated Metrics:")
for key, stat in stats.items():
    print(f"{key}: mean = {stat['mean']}, std = {stat['std']}")

# Plot the metrics
plot_metrics(all_metrics, 'train_loss')
plot_metrics(all_metrics, 'val_loss')
plot_metrics(all_metrics, 'test_loss')
plot_metrics(all_metrics, 'train_accuracy')
plot_metrics(all_metrics, 'val_accuracy')
plot_metrics(all_metrics, 'test_accuracy')












# from collections import defaultdict
# import os
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# def extract_metrics(log_dir, steps_per_epoch):
#     event_files = []
#     for root, _, files in os.walk(log_dir):
#         for file in files:
#             if 'events.out.tfevents' in file:
#                 event_files.append(os.path.join(root, file))

#     if not event_files:
#         print("No event files found.")
#         return

#     metrics = defaultdict(list)
#     for event_file in event_files:
#         event_acc = EventAccumulator(event_file)
#         event_acc.Reload()

#         tags = event_acc.Tags()
#         if 'scalars' not in tags:
#             continue

#         scalar_tags = tags['scalars']
#         for tag in scalar_tags:
#             events = event_acc.Scalars(tag)
#             values = [(e.step // steps_per_epoch, e.value) for e in events]
#             metrics[tag].extend(values)

#     return metrics

# def plot_metrics(metrics):
#     train_loss = metrics.get('train_loss', [])
#     val_loss = metrics.get('val_loss', [])
#     test_loss = metrics.get('test_loss', []) or metrics.get('test_test_loss', [])
#     train_accuracy = metrics.get('train_accuracy', [])
#     val_accuracy = metrics.get('val_accuracy', [])
#     test_accuracy = metrics.get('test_accuracy', []) or metrics.get('test_test_accuracy', [])

#     # Debugging prints
#     print("Train Loss:", train_loss)
#     print("Validation Loss:", val_loss)
#     print("Test Loss:", test_loss)
#     print("Train Accuracy:", train_accuracy)
#     print("Validation Accuracy:", val_accuracy)
#     print("Test Accuracy:", test_accuracy)

#     epochs_train_loss = [x[0] for x in train_loss]
#     values_train_loss = [x[1] for x in train_loss]
#     epochs_val_loss = [x[0] for x in val_loss]
#     values_val_loss = [x[1] for x in val_loss]
#     epochs_test_loss = [x[0] for x in test_loss]
#     values_test_loss = [x[1] for x in test_loss]

#     epochs_train_accuracy = [x[0] for x in train_accuracy]
#     values_train_accuracy = [x[1] for x in train_accuracy]
#     epochs_val_accuracy = [x[0] for x in val_accuracy]
#     values_val_accuracy = [x[1] for x in val_accuracy]
#     epochs_test_accuracy = [x[0] for x in test_accuracy]
#     values_test_accuracy = [x[1] for x in test_accuracy]

#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs_train_loss, values_train_loss, label='Train Loss')
#     plt.plot(epochs_val_loss, values_val_loss, label='Validation Loss')
#     if epochs_test_loss:
#         plt.scatter(epochs_test_loss, values_test_loss, label='Test Loss', color='green')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Curves')
#     plt.show()

#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs_train_accuracy, values_train_accuracy, label='Train Accuracy')
#     plt.plot(epochs_val_accuracy, values_val_accuracy, label='Validation Accuracy')
#     if epochs_test_accuracy:
#         plt.scatter(epochs_test_accuracy, values_test_accuracy, label='Test Accuracy', color='green')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.title('Accuracy Curves')
#     plt.show()

# log_dir = '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/AdamW_KD/Adagrad/distillation/Fine_Tuning/DeepShip/TDNN_CNN_14/Run_1/tb_logs/model_logs/run_1'

# steps_per_epoch = 39  # Adjust this value based on your actual steps per epoch
# metrics = extract_metrics(log_dir, steps_per_epoch)

# # # Print aggregated metrics
# # print("Aggregated Metrics:")
# # for key, values in metrics.items():
# #     avg_value = sum(value for _, value in values) / len(values)
# #     print(f"Aggregated {key}: {avg_value}")

# # Plot the metrics
# plot_metrics(metrics)
