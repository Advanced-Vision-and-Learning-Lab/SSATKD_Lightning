# import os
# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# def extract_metrics(log_dir):
#     event_files = []
#     for root, _, files in os.walk(log_dir):
#         for file in files:
#             if 'events.out.tfevents' in file:
#                 event_files.append(os.path.join(root, file))

#     if not event_files:
#         print(f"No event files found in {log_dir}.")
#         return {}

#     metrics = defaultdict(list)
#     for event_file in event_files:
#         event_acc = EventAccumulator(event_file)
#         event_acc.Reload()

#         tags = event_acc.Tags()
#         print(f"Tags for {event_file}: {tags}")  # Print the tags

#         if 'scalars' not in tags:
#             continue

#         scalar_tags = tags['scalars']
#         epochs = {}
#         for tag in scalar_tags:
#             events = event_acc.Scalars(tag)
#             if tag == 'epoch':
#                 epochs = {e.step: e.value for e in events}
#                 break

#         if not epochs:
#             print(f"No epoch tag found in {event_file}.")
#             continue

#         for tag in scalar_tags:
#             if tag != 'epoch':
#                 events = event_acc.Scalars(tag)
#                 for e in events:
#                     if e.step in epochs:
#                         epoch = epochs[e.step]
#                         metrics[tag].append((epoch, e.value))

#     return metrics

# def aggregate_metrics(runs_dirs):
#     all_metrics = defaultdict(list)
#     for run_dir in runs_dirs:
#         print(f"Checking directory: {run_dir}")
#         metrics = extract_metrics(run_dir)
#         if metrics:
#             for key, values in metrics.items():
#                 all_metrics[key].append(values)
#     return all_metrics

# def compute_stats(all_metrics):
#     stats = {}
#     for key, values_list in all_metrics.items():
#         all_values = [value for values in values_list for _, value in values]
#         if all_values:
#             mean = np.mean(all_values)
#             std = np.std(all_values)
#             stats[key] = {'mean': mean, 'std': std}
#     return stats

# def plot_metrics(all_metrics, metric_name):
#     plt.figure(figsize=(10, 5))
#     for run_idx, values in enumerate(all_metrics.get(metric_name, [])):
#         epochs = [epoch for epoch, _ in values]
#         values = [value for _, value in values]
#         plt.plot(epochs, values, label=f'Run {run_idx + 1}')

#     plt.xlabel('Epochs')
#     plt.ylabel(metric_name.replace('_', ' ').title())
#     plt.legend()
#     plt.title(f'{metric_name.replace("_", " ").title()} Across Runs')
#     plt.show()

# # Example usage:
# runs_dirs = [
#     '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/AdamW_withoutLR/Adagrad/distillation/Fine_Tuning/DeepShip/TDNN_CNN_14/Run_1/tb_logs/model_logs/run_1',
#     '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/AdamW_withoutLR/Adagrad/distillation/Fine_Tuning/DeepShip/TDNN_CNN_14/Run_2/tb_logs/model_logs/run_2',
#     '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/AdamW_withoutLR/Adagrad/distillation/Fine_Tuning/DeepShip/TDNN_CNN_14/Run_3/tb_logs/model_logs/run_3'
# ]

# all_metrics = aggregate_metrics(runs_dirs)
# stats = compute_stats(all_metrics)

# # Print detailed metrics for each run
# for metric_name, runs_values in all_metrics.items():
#     print(f"\nMetric: {metric_name}")
#     for run_idx, values in enumerate(runs_values):
#         print(f" Run {run_idx + 1}:")
#         for epoch, value in values:
#             print(f"  Epoch {epoch}: {value}")

# # Print aggregated statistics
# print("\nAggregated Metrics:")
# for key, stat in stats.items():
#     print(f"{key}: mean = {stat['mean']}, std = {stat['std']}")

# # Plot the metrics
# metrics_to_plot = [
#     'train_loss', 'val_loss', 'val_classification_loss', 'val_distillation_loss',
#     'val_struct_loss', 'val_stats_loss', 'test_loss', 'train_accuracy',
#     'val_accuracy', 'test_accuracy'
# ]

# for metric in metrics_to_plot:
#     plot_metrics(all_metrics, metric)




import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_metrics(log_dir):
    event_files = [os.path.join(root, file)
                   for root, _, files in os.walk(log_dir)
                   for file in files if 'events.out.tfevents' in file]

    if not event_files:
        print(f"No event files found in {log_dir}.")
        return {}

    metrics = defaultdict(list)
    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        try:
            event_acc.Reload()
        except Exception as e:
            print(f"Error loading {event_file}: {e}")
            continue

        tags = event_acc.Tags()
        # print(f"Tags for {event_file}: {tags}")

        if 'scalars' not in tags:
            continue

        scalar_tags = tags['scalars']
        epochs = {}
        for tag in scalar_tags:
            events = event_acc.Scalars(tag)
            if tag == 'epoch':
                epochs = {e.step: e.value for e in events}
                break

        if not epochs:
            print(f"No epoch tag found in {event_file}.")
            continue

        for tag in scalar_tags:
            if tag != 'epoch':
                events = event_acc.Scalars(tag)
                for e in events:
                    if e.step in epochs:
                        epoch = epochs[e.step]
                        metrics[tag].append((epoch, e.value))

    return metrics

def aggregate_metrics(runs_dirs):
    all_metrics = defaultdict(list)
    for run_dir in runs_dirs:
        # print(f"Checking directory: {run_dir}")
        metrics = extract_metrics(run_dir)
        if metrics:
            for key, values in metrics.items():
                all_metrics[key].append(values)
    return all_metrics

def compute_stats(all_metrics):
    stats = {}
    for key, values_list in all_metrics.items():
        all_values = [value for values in values_list for _, value in values]
        if all_values:
            mean = np.mean(all_values)
            std = np.std(all_values)
            stats[key] = {'mean': mean, 'std': std}
    return stats



def plot_metrics_for_each_run(all_metrics, metric_name_pairs, run_index):
    for train_metric_name, val_metric_name, title in metric_name_pairs:
        plt.figure(figsize=(10, 5))
        train_values = all_metrics.get(train_metric_name, [])[run_index]
        val_values = all_metrics.get(val_metric_name, [])[run_index]
        
        if not train_values or not val_values:
            print(f"No data for {train_metric_name} or {val_metric_name} in run {run_index + 1}")
            continue

        train_epochs = [epoch for epoch, _ in train_values]
        train_values = [value for _, value in train_values]
        val_epochs = [epoch for epoch, _ in val_values]
        val_values = [value for _, value in val_values]

        plt.plot(train_epochs, train_values, label='Train')
        plt.plot(val_epochs, val_values, label='Val')
        plt.xlabel('Epochs')
        plt.ylabel(title.replace('_', ' ').title())
        plt.legend()
        plt.title(f'{title.replace("_", " ").title()} for Run {run_index + 1}')
        plt.show()
        
# Example usage:
runs_dirs = [
    '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/confusion_bin8/AdamW/distillation/Fine_Tuning/DeepShip/TDNN_MobileNetV1/Run_1/tb_logs/model_logs/version_0',
    '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/confusion_bin8/AdamW/distillation/Fine_Tuning/DeepShip/TDNN_MobileNetV1/Run_2/tb_logs/model_logs/version_0',
    '/home/grads/j/jarin.ritu/Documents/Research/SSTKAD_Lightning/Saved_Models/confusion_bin8/AdamW/distillation/Fine_Tuning/DeepShip/TDNN_MobileNetV1/Run_3/tb_logs/model_logs/version_0'
]
all_metrics = aggregate_metrics(runs_dirs)
stats = compute_stats(all_metrics)


# Print aggregated statistics
print("\nAggregated Metrics:")
for key, stat in stats.items():
    print(f"{key}: mean = {stat['mean']}, std = {stat['std']}")

# Define metric name mappings for kd experiment
loss_metrics = [
    ('classification_loss', 'val_classification_loss', 'Classification Loss'),
    ('distillation_loss', 'val_distillation_loss', 'Distillation Loss'),
    ('struct_loss', 'val_struct_loss', 'Struct Loss'),
    ('stats_loss', 'val_stats_loss', 'Stats Loss')
]
# loss_metrics = [
#     ('train_loss', 'val_loss', 'Loss'),
#     ('train_accuracy', 'val_accuracy', 'Accuracy')
# ]
# Plot the metrics for each run separately
for run_index in range(len(runs_dirs)):
    plot_metrics_for_each_run(all_metrics, loss_metrics, run_index)






















