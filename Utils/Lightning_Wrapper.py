"""
Created on Thursday April 25 22:32:00 2024
Wrap models in a PyTorch Lightning Module for training and evaluation
@author: salimalkharsa
"""


# PyTorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as L
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import confusion_matrix
from Utils.Loss_function import Get_total_loss
import csv
import os
import pdb
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np


def plot_feature_maps(feature_maps, title):
    num_feature_maps = feature_maps.shape[1]
    size = feature_maps.shape[-1]
    cols = 8
    rows = num_feature_maps // cols + 1
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_feature_maps:
                ax = axs[i, j]
                feature_map = feature_maps[0, idx].cpu().detach().numpy()
                ax.imshow(feature_map, cmap='viridis')
                ax.axis('off')
            else:
                axs[i, j].axis('off')
    plt.suptitle(title)
    plt.show()
    
def plot_single_feature_map(feature_map, title, feature_map_index=0):
    fig, ax = plt.subplots(figsize=(5, 5))
    feature_map = feature_map[0, feature_map_index].cpu().detach().numpy()
    ax.imshow(feature_map, cmap='viridis')
    ax.axis('off')
    plt.suptitle(title)
    plt.show()
class Lightning_Wrapper(L.LightningModule):
    def __init__(self, model, num_classes, optimizer=optim.Adam, learning_rate=1e-3,
                 scheduler=None, criterion=nn.CrossEntropyLoss(), log_dir=None, 
                 label_names=None, stage='train',average='weighted'):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion','model'])
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.log_dir = log_dir
        self.stage = stage
        
        #Select average for f1, precision, and recall
        #Options are macro (sensitive to class imbalance), 
        # micro (sensitive to majority class), and weighted (balance between macro/micro)
        self.average = average
        
        #If names not provided, generate names (Class 1, ... , Class C)
        if label_names is None:
            self.label_names = []
            for class_name in range(0, self.num_classes):
                self.label_names.append('Class {}'.format(class_name))
        else:
            self.label_names = label_names
        
        #Change tasks based on number of classes (only consider binary and multiclass)
        if self.num_classes == 2:
            task = "binary"
        else:
            task = "multiclass"

        self.train_accuracy = torchmetrics.classification.Accuracy(task=task, num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task=task, num_classes=self.num_classes)
        self.val_f1 = torchmetrics.F1Score(task=task, num_classes=self.num_classes, average=average)
        self.val_precision = torchmetrics.Precision(task=task, num_classes=self.num_classes, average=average)
        self.val_recall = torchmetrics.Recall(task=task, num_classes=self.num_classes, average=average)

        self.test_accuracy = torchmetrics.classification.Accuracy(task=task, num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task=task, num_classes=self.num_classes, average=average)
        self.test_precision = torchmetrics.Precision(task=task, num_classes=self.num_classes, average=average)
        self.test_recall = torchmetrics.Recall(task=task, num_classes=self.num_classes, average=average)

        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []

    def forward(self, x):
        
        _, outputs = self.model(x)
        return outputs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        signals, labels, idx = batch
        _, outputs= self.model(signals) 
        loss = self.criterion(outputs, labels.long())

        accuracy = getattr(self, 'train_accuracy')(outputs, labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        signals, labels, idx = batch
        _, outputs= self.model(signals)
        loss = self.criterion(outputs, labels.long())
        accuracy = getattr(self, 'val_accuracy')(outputs, labels)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        outputs.argmax(dim=1).tolist()
        
        if self.stage == 'test':
            self.val_preds.extend(outputs.argmax(dim=1).tolist())
            self.val_labels.extend(labels.tolist())
            self.log_metrics(outputs, labels, prefix='val')

        return loss

    def test_step(self, batch, batch_idx):
        signals, labels, idx = batch
        _, outputs = self.model(signals)
        loss = self.criterion(outputs, labels.long())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.test_preds.extend(outputs.argmax(dim=1).tolist())
        self.test_labels.extend(labels.tolist())

        self.log_metrics(outputs, labels, prefix='test')

        return loss

    def log_metrics(self, outputs, labels, prefix):
        accuracy = getattr(self, f'{prefix}_accuracy')(outputs, labels)
        f1 = getattr(self, f'{prefix}_f1')(outputs, labels)
        precision = getattr(self, f'{prefix}_precision')(outputs, labels)
        recall = getattr(self, f'{prefix}_recall')(outputs, labels)

        self.log(f'{prefix}_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_f1', f1, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_precision', precision, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_recall', recall, on_step=False, on_epoch=True)

        # Log confusion matrix for validation and test sets
        if prefix in ['val', 'test']:
            preds = outputs.argmax(dim=1).tolist()
            # Change the labels here to be the actual labels
            label_names = self.label_names

            # Change the preds here to be the predicted labels with the label names in the same structure as the labels
            preds = [label_names[x] for x in preds]
            # Change the true label names to be the actual labels
            labels = [label_names[x] for x in labels.tolist()]
            
            # Compute confusion matrix, including all labels
            cm = confusion_matrix(labels, preds, labels=label_names)
        self.log_confusion_matrix(cm, prefix)

    def on_validation_epoch_end(self):
        if self.val_preds and self.val_labels:
            cm = confusion_matrix(self.val_labels, self.val_preds, labels=range(self.num_classes))
            self.log_confusion_matrix(cm, 'val')
            self.val_preds.clear()
            self.val_labels.clear()

    def on_test_epoch_end(self):
        if self.test_preds and self.test_labels:
            cm = confusion_matrix(self.test_labels, self.test_preds, labels=range(self.num_classes))
            self.log_confusion_matrix(cm, 'test')
            self.test_preds.clear()
            self.test_labels.clear()

    def log_confusion_matrix(self, cm, prefix):
        # Save confusion matrix to CSV file in log directory
        csv_file = os.path.join(self.log_dir, f'{prefix}_confusion_matrix.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write confusion matrix to CSV row by row
            writer.writerow([''] + self.label_names)  # Header row
            for i, row in enumerate(cm):
                writer.writerow([self.label_names[i]] + row.tolist())



class Lightning_Wrapper_KD(L.LightningModule):
    def __init__(self, model,num_classes, stats_w, struct_w, distill_w, optimizer=optim.Adam, learning_rate=1e-3,
                 scheduler=None, criterion=nn.CrossEntropyLoss(), log_dir=None, 
                 label_names=None, stage='train',average='weighted', Params=None):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion','model'])
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.log_dir = log_dir
        self.stage = stage
        self.stats_w = stats_w
        self.distill_w = distill_w
        self.struct_w = struct_w
        
        #Select average for f1, precision, and recall
        #Options are macro (sensitive to class imbalance), 
        # micro (sensitive to majority class), and weighted (balance between macro/micro)
        self.average = average
        
        #If names not provided, generate names (Class 1, ... , Class C)
        if label_names is None:
            self.label_names = []
            for class_name in range(0, self.num_classes):
                self.label_names.append('Class {}'.format(class_name))
        else:
            self.label_names = label_names
        
        #Change tasks based on number of classes (only consider binary and multiclass)
        if self.num_classes == 2:
            task = "binary"
        else:
            task = "multiclass"

        self.train_accuracy = torchmetrics.classification.Accuracy(task=task, num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task=task, num_classes=self.num_classes)
        self.val_f1 = torchmetrics.F1Score(task=task, num_classes=self.num_classes, average=average)
        self.val_precision = torchmetrics.Precision(task=task, num_classes=self.num_classes, average=average)
        self.val_recall = torchmetrics.Recall(task=task, num_classes=self.num_classes, average=average)

        self.test_accuracy = torchmetrics.classification.Accuracy(task=task, num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task=task, num_classes=self.num_classes, average=average)
        self.test_precision = torchmetrics.Precision(task=task, num_classes=self.num_classes, average=average)
        self.test_recall = torchmetrics.Recall(task=task, num_classes=self.num_classes, average=average)

        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []
        
        # # Knowledge distillation parameters
        # self.temperature = Params['temperature']
        # self.alpha = Params['alpha']
        
    def forward(self, x):
        # pdb.set_trace()
       
        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(x)
        
        return struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher
    
    def print_trainable_parameters(model):
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, Shape: {param.shape}, Number of elements: {param.numel()}")
    
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        signals, labels, idx = batch
        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(signals)
        # pdb.set_trace()
            # with torch.profiler.record_function("calculate_loss"):
        # classification_loss = F.cross_entropy(output_student, labels)
        # loss, classification_loss, distillation_loss, struct_loss, stats_loss = Get_total_loss(
        #     struct_feats_teacher, struct_feats_student, stats_feats_teacher, stats_feats_student, 
        #     output_teacher, output_student, classification_loss
        #         )
        loss, loss_dict = self.criterion(struct_feats_teacher, struct_feats_student, 
                                         stats_feats_teacher, stats_feats_student,
                                         output_teacher, output_student,labels)


        accuracy = getattr(self, 'train_accuracy')(output_student, labels)

        # Log accuracy and loss
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        # self.log('classification_loss', classification_loss, on_step=False, on_epoch=True)
        # self.log('distillation_loss', distillation_loss, on_step=False, on_epoch=True)
        # self.log('struct_loss', struct_loss, on_step=False, on_epoch=True)
        # self.log('stats_loss', stats_loss, on_step=False, on_epoch=True)
        
        self.log('classification_loss', loss_dict['class_loss'], on_step=False, on_epoch=True)
        self.log('distillation_loss', loss_dict['distill_loss'], on_step=False, on_epoch=True)
        self.log('struct_loss', loss_dict['struct_loss'], on_step=False, on_epoch=True)
        self.log('stats_loss', loss_dict['stat_loss'], on_step=False, on_epoch=True)
        
        # # # Print profiling results
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        # plot_single_feature_map(struct_feats_teacher, "Structural Features Teacher", feature_map_index=0)
        # plot_single_feature_map(stats_feats_teacher, "Statistical Features Teacher", feature_map_index=0)
    
        return loss



    def validation_step(self, batch, batch_idx):
        signals, labels, idx = batch
     
        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(signals)    
        # pdb.set_trace()        
        # classification_loss = F.cross_entropy(output_student, labels)
        # loss, classification_loss, distillation_loss, struct_loss, stats_loss = Get_total_loss(
        #     struct_feats_teacher, struct_feats_student, stats_feats_teacher, stats_feats_student, 
        #     output_teacher, output_student, classification_loss
        # )
        
        loss, loss_dict = self.criterion(struct_feats_teacher, struct_feats_student, 
                                         stats_feats_teacher, stats_feats_student,
                                         output_teacher, output_student,labels)
     
        accuracy = getattr(self, 'val_accuracy')(output_student, labels)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', loss.mean(), on_step=False, on_epoch=True)
        self.log('val_classification_loss', loss_dict['class_loss'], on_step=False, on_epoch=True)
        self.log('val_distillation_loss', loss_dict['distill_loss'], on_step=False, on_epoch=True)
        self.log('val_struct_loss', loss_dict['struct_loss'], on_step=False, on_epoch=True)
        self.log('val_stats_loss', loss_dict['stat_loss'], on_step=False, on_epoch=True)
        # self.log('val_classification_loss', classification_loss, on_step=False, on_epoch=True)
        # self.log('val_distillation_loss', distillation_loss, on_step=False, on_epoch=True)
        # self.log('val_struct_loss', struct_loss, on_step=False, on_epoch=True)
        # self.log('val_stats_loss', stats_loss, on_step=False, on_epoch=True)
     
        # Print profiling results
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
        if self.stage == 'test':
            self.val_preds.extend(output_student.argmax(dim=1).tolist())
            self.val_labels.extend(labels.tolist())
            self.log_metrics(output_student, labels, prefix='val')
     
        return loss


    def test_step(self, batch, batch_idx):
        signals, labels, idx = batch
    
        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(signals)
            
        loss, _ = self.criterion(output_student, labels.long())
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.test_preds.extend(output_student.argmax(dim=1).tolist())
        self.test_labels.extend(labels.tolist())
    
        self.log_metrics(output_student, labels, prefix='test')
    

    
        return loss


    def log_metrics(self, outputs, labels, prefix):
        accuracy = getattr(self, f'{prefix}_accuracy')(outputs, labels)
        f1 = getattr(self, f'{prefix}_f1')(outputs, labels)
        precision = getattr(self, f'{prefix}_precision')(outputs, labels)
        recall = getattr(self, f'{prefix}_recall')(outputs, labels)

        self.log(f'{prefix}_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_f1', f1, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_precision', precision, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_recall', recall, on_step=False, on_epoch=True)

        # Log confusion matrix for validation and test sets
        if prefix in ['val', 'test']:
            preds = outputs.argmax(dim=1).tolist()
            # Change the labels here to be the actual labels
            label_names = self.label_names

            # Change the preds here to be the predicted labels with the label names in the same structure as the labels
            preds = [label_names[x] for x in preds]
            # Change the true label names to be the actual labels
            labels = [label_names[x] for x in labels.tolist()]
            
            # Compute confusion matrix, including all labels
            cm = confusion_matrix(labels, preds, labels=label_names)
        self.log_confusion_matrix(cm, prefix)

    def on_validation_epoch_end(self):
        if self.val_preds and self.val_labels:
            cm = confusion_matrix(self.val_labels, self.val_preds, labels=range(self.num_classes))
            self.log_confusion_matrix(cm, 'val')
            self.val_preds.clear()
            self.val_labels.clear()

    def on_test_epoch_end(self):
        if self.test_preds and self.test_labels:
            cm = confusion_matrix(self.test_labels, self.test_preds, labels=range(self.num_classes))
            self.log_confusion_matrix(cm, 'test')
            self.test_preds.clear()
            self.test_labels.clear()

    def log_confusion_matrix(self, cm, prefix):
        # Save confusion matrix to CSV file in log directory
        csv_file = os.path.join(self.log_dir, f'{prefix}_confusion_matrix.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write confusion matrix to CSV row by row
            writer.writerow([''] + self.label_names)  # Header row
            for i, row in enumerate(cm):
                writer.writerow([self.label_names[i]] + row.tolist())

