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
import seaborn as sns
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from Utils.CustomLRScheduler import CustomLRScheduler
from lightning.pytorch.utilities import grad_norm
torch.autograd.set_detect_anomaly(True)
from Utils.Loss_function import SSTKAD_Loss
class Lightning_Wrapper(L.LightningModule):
    def __init__(self, model, num_classes, max_iter,optimizer=optim.AdamW, lr=0.015,
                 scheduler=None, criterion=nn.CrossEntropyLoss(), log_dir=None, 
                 label_names=None, stage='train',average='weighted'):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion','model'])
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.log_dir = log_dir
        self.stage = stage
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.max_iter = max_iter
        
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

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
    #     scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.max_iter, power=0.9, last_epoch=-1)
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.max_iter, power=0.9, last_epoch=-1)
        return [optimizer], [scheduler]
    
    # def configure_optimizers(self):
    #     optimizer = self.optimizer(self.parameters(), lr=self.lr)
    #     return optimizer

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
        print("\nVal Loss",loss)
        print("\nVal acc",accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        # pdb.set_trace()
        signals, labels, idx = batch
        _, outputs = self.model(signals)
        loss = self.criterion(outputs, labels.long())
        # accuracy = getattr(self, 'test_accuracy')(outputs, labels)


        self.log('test_loss', loss, on_step=False, on_epoch=True)
        # self.log('test_accuracy',accuracy, on_step=False, on_epoch=True)
        self.test_preds.extend(outputs.argmax(dim=1).tolist())
        self.test_labels.extend(labels.tolist())
        

        self.log_metrics(outputs, labels, prefix='test')

        return loss

    def log_metrics(self, outputs, labels, prefix):
        accuracy = getattr(self, f'{prefix}_accuracy')(outputs, labels)
        print("\nTest acc", accuracy)
        f1 = getattr(self, f'{prefix}_f1')(outputs, labels)
        precision = getattr(self, f'{prefix}_precision')(outputs, labels)
        recall = getattr(self, f'{prefix}_recall')(outputs, labels)

        self.log(f'{prefix}_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_f1', f1, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_precision', precision, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_recall', recall, on_step=False, on_epoch=True)
        # pdb.set_trace()
        # # Log confusion matrix for validation and test sets
        # if prefix in ['val', 'test']:
        #     preds = outputs.argmax(dim=1).tolist()
        #     # Change the labels here to be the actual labels
        #     label_names = self.label_names

        #     # Change the preds here to be the predicted labels with the label names in the same structure as the labels
        #     preds = [label_names[x] for x in preds]
        #     # Change the true label names to be the actual labels
        #     labels = [label_names[x] for x in labels.tolist()]
            
        #     # Compute confusion matrix, including all labels
        #     cm = confusion_matrix(labels, preds, labels=label_names)
        # self.log_confusion_matrix(cm, prefix)

    def on_validation_epoch_end(self):
        if self.val_preds and self.val_labels:
            cm = confusion_matrix(self.val_labels, self.val_preds, labels=range(self.num_classes))
            self.log_confusion_matrix(cm, 'val')
            self.val_preds.clear()
            self.val_labels.clear()

    def on_test_epoch_end(self):
        if self.test_preds and self.test_labels:
            cm = confusion_matrix(self.test_labels, self.test_preds, labels=range(self.num_classes))
            # self.log_confusion_matrix(cm, 'test')
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
    def __init__(self, model,num_classes, stats_w, struct_w, distill_w, max_iter,optimizer=optim.AdamW, lr=0.015,
                 scheduler=None,  criterion=SSTKAD_Loss(task_num = 4), log_dir=None, 
                 label_names=None, stage='train',average='weighted', Params=None):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion','model'])
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.log_dir = log_dir
        self.stage = stage
        self.stats_w = stats_w
        self.distill_w = distill_w
        self.struct_w = struct_w
        self.val_preds = []
        self.val_labels = []
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.max_iter = max_iter
        
        
        print("Learning rate: ", self.lr)
        
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
        
        self.quant_student_list = []
        self.quant_teacher_list = []
        
        self.quant_student_trained = None
        self.quant_teacher_trained = None
        

        
    def forward(self, x):
        # pdb.set_trace()
       
        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(x)
        
        return struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher
    
    def on_fit_start(self):
        if self.lr:
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lr
    # def print_trainable_parameters(model):
    #     print("Trainable parameters:")
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"Parameter: {name}, Shape: {param.shape}, Number of elements: {param.numel()}")
    
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=0.015,momentum=self.momentum, weight_decay=self.weight_decay)
    #     scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.max_iter, power=0.9, last_epoch=-1)
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.max_iter, power=0.9, last_epoch=-1)
        return [optimizer], [scheduler]
    

    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        signals, labels, idx = batch


        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(signals)

        loss, loss_dict = self.criterion(struct_feats_teacher, struct_feats_student, 
                                         stats_feats_teacher, stats_feats_student,
                                         output_teacher, output_student,labels)
        # pdb.set_trace()
        # print("\nTrain loss",loss)
        # if torch.isnan(loss):
        #     pdb.set_trace()

        accuracy = getattr(self, 'train_accuracy')(output_student, labels)
        # print("\nTrain Loss",loss)
        # print("\nTrain acc",accuracy)

        # Log accuracy and loss
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        
        self.log('classification_loss', loss_dict['class_loss'], on_step=False, on_epoch=True)
        self.log('distillation_loss', loss_dict['distill_loss'], on_step=False, on_epoch=True)
        self.log('struct_loss', loss_dict['struct_loss'], on_step=False, on_epoch=True)
        self.log('stats_loss', loss_dict['stat_loss'], on_step=False, on_epoch=True)
        self.log('learning_rate', self.lr, on_step=False, on_epoch=True)
        
        # # Squeeze and store the quantities
        # quant_student = quant_student.squeeze(1).cpu().detach().numpy()
        # quant_teacher = quant_teacher.squeeze(1).cpu().detach().numpy()
        
        # fig, ax = plt.subplots(8,8)
        # test = quant_student[0]
        
        # for i in range(0,8):
        #     for j in range(0,8):
        #         ax[i,j].imshow(test[i,j])
                
        
        # plt.show()


        # # Store only the first quant matrix of the batch
        # self.quant_student_list.append(quant_student[0])
        # self.quant_teacher_list.append(quant_teacher[0])

        return loss




    def validation_step(self, batch, batch_idx):
        signals, labels, idx = batch
    
        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher = self.model(signals)
    
        loss, loss_dict = self.criterion(struct_feats_teacher, struct_feats_student, 
                                         stats_feats_teacher, stats_feats_student,
                                         output_teacher, output_student, labels)
        # print("\nVal loss", loss)
        # if torch.isnan(loss):
        #     pdb.set_trace()
    
        accuracy = getattr(self, 'val_accuracy')(output_student, labels)
        # pdb.set_trace()
        # print("\nVal acc", accuracy)
        if torch.isnan(accuracy):
            print("\nVal acc", accuracy)
            # pdb.set_trace()
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_classification_loss', loss_dict['class_loss'], on_step=False, on_epoch=True)
        self.log('val_distillation_loss', loss_dict['distill_loss'], on_step=False, on_epoch=True)
        self.log('val_struct_loss', loss_dict['struct_loss'], on_step=False, on_epoch=True)
        self.log('val_stats_loss', loss_dict['stat_loss'], on_step=False, on_epoch=True)
        if self.stage == 'test':
            self.val_preds.extend(output_student.argmax(dim=1).tolist())
            self.val_labels.extend(labels.tolist())
            self.log_metrics(output_student, labels, prefix='val')
        # print("\nVal Loss",loss)
        # print("\nVal acc",accuracy)
        return loss
        # return {'loss': loss, 'preds': preds, 'labels': labels}
        
    


    def test_step(self, batch, batch_idx):
        # pdb.set_trace()
        signals, labels, idx = batch
    
        struct_feats_student, struct_feats_teacher, stats_feats_student, stats_feats_teacher, output_student, output_teacher  = self.model(signals)
    
        loss, _ = self.criterion(struct_feats_teacher, struct_feats_student, 
                                 stats_feats_teacher, stats_feats_student,
                                 output_teacher, output_student, labels)
    
        self.log('test_loss', loss, on_step=False, on_epoch=True)
    
        preds = output_student.argmax(dim=1)
        self.test_preds.extend(preds.tolist())
        self.test_labels.extend(labels.tolist())
    
        self.log_metrics(output_student, labels, prefix='test')
    
        return loss

    def log_metrics(self, outputs, labels, prefix):
        # pdb.set_trace()
        accuracy = getattr(self, f'{prefix}_accuracy')(outputs, labels)
        f1 = getattr(self, f'{prefix}_f1')(outputs, labels)
        precision = getattr(self, f'{prefix}_precision')(outputs, labels)
        recall = getattr(self, f'{prefix}_recall')(outputs, labels)

        self.log(f'{prefix}_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_f1', f1, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_precision', precision, on_step=False, on_epoch=True)
        self.log(f'{prefix}_{self.average}_recall', recall, on_step=False, on_epoch=True)


    
    def on_validation_epoch_end(self):
        if self.val_preds and self.val_labels:
            # cm = confusion_matrix(self.val_labels, self.val_preds, labels=range(self.num_classes))
            # self.plot_confusion_matrix(cm, 'val')
            self.val_preds.clear()
            self.val_labels.clear()


    def log_confusion_matrix(self, cm, prefix):
        # Save confusion matrix to CSV file in log directory
        csv_file = os.path.join(self.log_dir, f'{prefix}_confusion_matrix.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write confusion matrix to CSV row by row
            writer.writerow([''] + self.label_names)  # Header row
            for i, row in enumerate(cm):
                writer.writerow([self.label_names[i]] + row.tolist())

    def on_test_epoch_end(self):
        if self.test_preds and self.test_labels:
            # cm = confusion_matrix(self.test_labels, self.test_preds, labels=range(self.num_classes))
            # self.plot_confusion_matrix(cm, 'test')
            self.test_preds.clear()
            self.test_labels.clear()

    def plot_confusion_matrix(self, cm, prefix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_names, yticklabels=self.label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{prefix.capitalize()} Confusion Matrix')
        
        # Save the figure
        fig_path = os.path.join(self.log_dir, f'{prefix}_confusion_matrix.png')
        plt.savefig(fig_path)
        plt.close()
        
        # Optionally, log the figure with the experiment logging framework
        if self.logger:
            self.logger.experiment.add_figure(f'{prefix}_confusion_matrix', plt.gcf(), self.current_epoch)
    # def on_train_end(self):
    #     # Ensure we have enough matrices to plot an 8x8 grid
    #     num_matrices = min(64, len(self.quant_student_list), len(self.quant_teacher_list))

    #     if num_matrices > 0:
    #         fig, ax = plt.subplots(8, 8, figsize=(20, 20))
    #         fig.suptitle('Quant Matrices Student')

    #         for i in range(8):
    #             for j in range(8):
    #                 index = i * 8 + j
    #                 if index < num_matrices:
    #                     ax[i, j].imshow(self.quant_student_list[index], cmap='viridis')
    #                     ax[i, j].axis('off')
    #                 else:
    #                     ax[i, j].axis('off')

    #         plt.show()

    #         fig, ax = plt.subplots(8, 8, figsize=(20, 20))
    #         fig.suptitle('Quant Matrices Teacher')

    #         for i in range(8):
    #             for j in range(8):
    #                 index = i * 8 + j
    #                 if index < num_matrices:
    #                     ax[i, j].imshow(self.quant_teacher_list[index], cmap='viridis')
    #                     ax[i, j].axis('off')
    #                 else:
    #                     ax[i, j].axis('off')

    #         plt.show()
    #         pdb.set_trace()