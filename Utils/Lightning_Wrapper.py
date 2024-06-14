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
        signals, labels, idx = batch
        _, outputs = self.model(signals)
        loss = self.criterion(outputs, labels.long())

        accuracy = getattr(self, 'train_accuracy')(outputs, labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        signals, labels, idx = batch
        _, outputs = self.model(signals)
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
    def __init__(self, model,num_classes, optimizer=optim.Adam, learning_rate=1e-3,
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
        
        # Knowledge distillation parameters
        self.temperature = Params['temperature']
        self.alpha = Params['alpha']
        
    def forward(self, x):
       
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
        
        #Return dictionary of outputs
        #outputs = self.model(signals)
        
        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(signals)
        classification_loss = F.cross_entropy(output_student, labels)
        
        loss, classification_loss, distillation_loss,struct_loss, stats_loss = Get_total_loss(struct_feats_teacher, struct_feats_student, stats_feats_teacher, stats_feats_student, 
                                                                                                  output_teacher, output_student, classification_loss)                  
     
        accuracy = getattr(self, 'train_accuracy')(output_student, labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        signals, labels, idx = batch

        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(signals)

        loss = self.criterion(output_student, labels.long())
        accuracy = getattr(self, 'val_accuracy')(output_student, labels)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        output_student.argmax(dim=1).tolist()
        
        if self.stage == 'test':
            self.val_preds.extend(output_student.argmax(dim=1).tolist())
            self.val_labels.extend(labels.tolist())
            self.log_metrics(output_student, labels, prefix='val')

        return loss

    def test_step(self, batch, batch_idx):
        signals, labels, idx = batch
        features = self.feature_layer(signals)
        _, outputs = self.model(features)
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

