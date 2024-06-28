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

import torch
torch.cuda.empty_cache()

# Set the device (use 'cuda' if a GPU is available, otherwise 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SSTKAD_Loss(nn.Module):
    def __init__(self):
        
        super(SSTKAD_Loss, self).__init__()
        
        #Initalize loss modules
        self.classification_loss = nn.CrossEntropyLoss()
        self.stats_loss = EarthMoversDistanceLoss()
        self.struct_loss = nn.CosineEmbeddingLoss()
        self.distill_loss = EarthMoversDistanceLoss()
        
        #Add learnable weights for loss function
        # self.weights = nn.Parameter(torch.ones(4)/4)
        self.weights = [.9, .7, .5]
        
        
    def forward(self,struct_teacher, struct_student, stats_teacher, stats_student, prob_teacher, prob_student, labels):
        
        #Compute each loss
        class_loss = self.classification_loss(prob_student,labels)
        
        #for statistical loss, use 2D EMD loss
        stat_loss = self.stats_loss(stats_teacher, stats_student)
        
        #For structural loss, use cosine embedding with target values equal to 1
        target_struct = torch.ones(struct_student.size(0)).to(struct_student.device)
        struct_loss = self.struct_loss(struct_student.flatten(1), struct_teacher.flatten(1), target_struct)
        
        #Compute distillation loss 
        prob_student = F.softmax(prob_student / 2, dim=-1)
        prob_teacher = F.softmax(prob_teacher / 2, dim=-1)
        distillation_loss = self.distill_loss(prob_student, prob_teacher)
        
        #Compute loss with weights
        loss = class_loss + (self.weights[0])*stat_loss + (self.weights[1])*struct_loss + (self.weights[2])*distillation_loss
        # loss = class_loss + (stats_w*stat_loss) + (struct_w*struct_loss) + (distill_w*distillation_loss)
        # print(self.weights)
        
        #Create dictionary for plotting
        loss_dict = {'class_loss': class_loss, 'stat_loss': stat_loss, 
                     'struct_loss': struct_loss, 'distill_loss': distillation_loss}
        
        return loss, loss_dict

def Get_total_loss(struct_teacher, struct_student, stats_teacher, stats_student, prob_teacher, prob_student, classification_loss):


    cosine_loss = nn.CosineEmbeddingLoss()
    
    # Flatten the spatial dimensions and channels to get (N, D) 
    target_struct = torch.ones(struct_student.size(0)).to(struct_student.device)
    
    # Compute the cosine loss
    struct_loss = cosine_loss(struct_student.flatten(1), struct_teacher.flatten(1), target_struct)
    # print("Cosine Loss:", struct_loss.item())


    
    emd_loss = EMDLoss2D()
    
    stats_loss = emd_loss (stats_teacher, stats_student)
        
    # target_stats = torch.ones(stats_student.size(0)).to(struct_student.device)
    # # Compute the cosine loss
    # stats_loss = cosine_loss(stats_student.flatten(1), stats_teacher.flatten(1), target_stats)

    # stats_loss = F.mse_loss(stats_student, stats_teacher, reduction='mean')
    print("Stats Loss:", stats_loss)

    dist_loss = EarthMoversDistanceLoss()
    # pdb.set_trace()

    softened_student_outputs = F.softmax(prob_student / 2, dim=-1)
    softened_teacher_outputs = F.softmax(prob_teacher / 2, dim=-1)
    #distillation_loss = F.kl_div(softened_student_outputs, softened_teacher_outputs.detach(), reduction='batchmean')
    distillation_loss = dist_loss(softened_student_outputs, softened_teacher_outputs)
    print("Probability Loss:", distillation_loss.item())

    


    # Combine all losses
    loss = ( classification_loss) + (0.9 * struct_loss) +  (0.7 * stats_loss) + (0.5 * distillation_loss)
    # print("Total Loss:", loss.item())

    return loss, classification_loss, distillation_loss, struct_loss, stats_loss




