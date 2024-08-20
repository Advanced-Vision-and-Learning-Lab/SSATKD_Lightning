#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:08:11 2024

@author: jarin.ritu
"""
import torch.nn.functional as F
import pdb
import torch.nn as nn
from Utils.EMD_loss import EarthMoversDistanceLoss, MutualInformationLoss, EMDLoss2D
import numpy as np

import torch
torch.cuda.empty_cache()

# Set the device (use 'cuda' if a GPU is available, otherwise 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SSTKAD_Loss(nn.Module):
    def __init__(self, task_num,task_flag=[True, True, True, True]):
        super(SSTKAD_Loss, self).__init__()
        
        # Initialize loss modules
        self.classification_loss = nn.CrossEntropyLoss()
        self.stats_loss = EMDLoss2D()
        self.struct_loss = nn.CosineEmbeddingLoss()
        self.distill_loss = EarthMoversDistanceLoss()
        # self.temperature = 1
        self.T_prob_student = nn.Parameter(torch.rand(1) * 15)  
        self.T_prob_teacher = nn.Parameter(torch.rand(1) * 15)  

        
        # Initialize learnable log variances for each task
        self.log_vars = nn.Parameter(torch.zeros(task_num))
        # self.task_flag = task_flag
        
        #For ablation study, set desired losses to be used (1) and others not used (0)
        self.log_vars_flag = np.array(task_flag).astype(float)
        
        
    def forward(self, struct_teacher, struct_student, stats_teacher, stats_student, prob_teacher, prob_student, labels):
        
   
        # Class loss - cross entrophy
        class_loss = self.classification_loss(prob_student, labels)
        
        # Statistical loss - 2D EMD loss
        stat_loss = self.stats_loss(stats_teacher, stats_student)
        
        # Structural loss - cosine embedding with target values equal to 1
        target_struct = torch.ones(struct_student.size(0)).to(struct_student.device)
        struct_loss = self.struct_loss(struct_student.flatten(1), struct_teacher.flatten(1), target_struct)
        
        
        # Distillation loss - EMD loss
        # prob_student_max,_ = prob_student.max(dim=1, keepdim=True)
        # prob_teacher_max,_ = prob_teacher.max(dim=1, keepdim=True)
        
        # T_prob_student = 2 *(prob_student_max) / (prob_teacher_max + prob_student_max) * self.temperature
        # T_prob_teacher = 2 *(prob_teacher_max) / (prob_teacher_max + prob_student_max) * self.temperature
            
        # prob_student = F.softmax(prob_student /T_prob_student, dim=1)
        # prob_teacher = F.softmax(prob_teacher /T_prob_teacher, dim=1)    
        # distill_loss = self.distill_loss(prob_student, prob_teacher,T_prob_student,T_prob_teacher)
        
        
        prob_student = F.softmax(prob_student /self.T_prob_student, dim=-1)
        prob_teacher = F.softmax(prob_teacher /self.T_prob_teacher, dim=-1)    
        distill_loss = self.distill_loss(prob_student, prob_teacher)
        
        
        
        
        # prob_student = F.softmax(prob_student /self.temperature, dim=-1)
        # prob_teacher = F.softmax(prob_teacher /self.temperature, dim=-1)    
        # distill_loss = self.distill_loss(prob_student, prob_teacher)
        
        
        
        #Ensure losses remain positive to minimize metrics 
        # Adjust losses with dynamic weights based on log variances
        precision_class = torch.exp(-self.log_vars[0])
        class_loss = precision_class * class_loss + (self.log_vars[0])
        
        precision_stat = torch.exp(-self.log_vars[1])
        stat_loss = precision_stat * stat_loss + (self.log_vars[1])

        precision_struct = torch.exp(-self.log_vars[2])
        struct_loss = precision_struct * struct_loss + (self.log_vars[2])

        precision_distill = torch.exp(-self.log_vars[3])
        distill_loss = precision_distill * distill_loss + (self.log_vars[3])

        

        # pdb.set_trace()
        # Compute total loss
        total_loss = (self.log_vars_flag[0] * class_loss) + (self.log_vars_flag[1] * stat_loss)
        + (self.log_vars_flag[2] * struct_loss) + (self.log_vars_flag[3] * distill_loss)
        
        # total_loss = (0.6*class_loss) + (0.5*struct_loss) + (0.4*stat_loss) + (0.9*distill_loss)
        
        # total_loss =  class_loss +  distill_loss
        # total_loss = (class_loss) 
        # print("\ndist loss",distill_loss)
        # Create dictionary for plotting
        loss_dict = {
            'class_loss': class_loss,
            'stat_loss': stat_loss,
            'struct_loss': struct_loss,
            'distill_loss': distill_loss
        }
        
        return total_loss, loss_dict

# class SSTKAD_Loss(nn.Module):
#     def __init__(self, task_num,task_flag=[True, True, True, True]):
#         super(SSTKAD_Loss, self).__init__()
        
#         # Initialize loss modules
#         self.classification_loss = nn.CrossEntropyLoss()
#         self.stats_loss = EMDLoss2D()
#         self.struct_loss = nn.CosineEmbeddingLoss()
#         self.distill_loss = EarthMoversDistanceLoss()
#         self.T_prob_student = nn.Parameter(torch.rand(1) * 15)  
#         self.T_prob_teacher = nn.Parameter(torch.rand(1) * 15)  
        
#         # Initialize learnable log variances for each task
#         self.log_vars = nn.Parameter(torch.zeros(task_num))
#         # self.task_flag = task_flag
        
#         #For ablation study, set desired losses to be used (1) and others not used (0)
#         self.log_vars_flag = np.array(task_flag).astype(float)

#     def forward(self, struct_teacher, struct_student, stats_teacher, stats_student, prob_teacher, prob_student, labels):
        
#         # Class loss - cross entrophy
#         class_loss = self.classification_loss(prob_student, labels)
        
#         # Statistical loss - 2D EMD loss
#         stat_loss = self.stats_loss(stats_teacher, stats_student)
        
#         # Structural loss - cosine embedding with target values equal to 1
#         target_struct = torch.ones(struct_student.size(0)).to(struct_student.device)
#         struct_loss = self.struct_loss(struct_student.flatten(1), struct_teacher.flatten(1), target_struct)
        
#         # Distillation loss - EMD loss
#         prob_student = F.softmax(prob_student / self.T_prob_student, dim=-1)
#         prob_teacher = F.softmax(prob_teacher / self.T_prob_teacher, dim=-1)
#         distill_loss = self.distill_loss(prob_student, prob_teacher)
        
#         #Ensure losses remain positive to minimize metrics 
    
#         # Adjust losses with dynamic weights based on log variances
#         # pdb.set_trace()
#         precision_class = torch.exp(-self.log_vars[0])
#         class_loss = precision_class * class_loss + F.relu(self.log_vars[0])
        
#         precision_stat = torch.exp(-self.log_vars[1])
#         stat_loss = precision_stat * stat_loss + F.relu(self.log_vars[1])

#         precision_struct = torch.exp(-self.log_vars[2])
#         struct_loss = precision_struct * struct_loss + F.relu(self.log_vars[2])

#         precision_distill = torch.exp(-self.log_vars[3])
#         distill_loss = precision_distill * distill_loss + F.relu(self.log_vars[3])

        
#         # print("\nClassification loss",class_loss)
#         # print("\nDistillation loss",distill_loss)

#         # Compute total loss
#         total_loss = (self.log_vars_flag[0] * class_loss) + (self.log_vars_flag[1] * stat_loss)
#         + (self.log_vars_flag[2] * struct_loss) + (self.log_vars_flag[3] * distill_loss)
#         # total_loss = (class_loss) 
#         # print("\ndist loss",distill_loss)
#         # Create dictionary for plotting
#         loss_dict = {
#             'class_loss': class_loss,
#             'stat_loss': stat_loss,
#             'struct_loss': struct_loss,
#             'distill_loss': distill_loss
#         }
        
#         return total_loss, loss_dict

