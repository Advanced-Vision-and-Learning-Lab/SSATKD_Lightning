#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:08:11 2024

@author: jarin.ritu
"""
import torch.nn.functional as F
import pdb

import torch
torch.cuda.empty_cache()

def Get_total_loss(struct_teacher, struct_student, stats_teacher, stats_student, prob_teacher, prob_student, classification_loss):
    struct_loss = F.mse_loss(struct_student, struct_teacher, reduction='mean')

    

    

    stats_loss = F.mse_loss(stats_student, stats_teacher, reduction='mean')

   

    softened_student_outputs = F.log_softmax(prob_student / 2, dim=1)
    softened_teacher_outputs = F.softmax(prob_teacher / 2, dim=1)
    distillation_loss = F.kl_div(softened_student_outputs, softened_teacher_outputs.detach(), reduction='batchmean')

    
    

    classification_loss = (classification_loss) 


    # print("classification_loss", classification_loss)
    # print("normalized_distillation_loss", distillation_loss)
    # print("struct_loss", struct_loss)
    # print("stats_loss", stats_loss)
    
    # Combine all losses
    loss = (0.9 * classification_loss) + (0.3 * struct_loss) +  (0.3 * stats_loss) + (0.4 * distillation_loss)

    return loss, classification_loss, distillation_loss, struct_loss, stats_loss
