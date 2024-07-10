#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:03:48 2024

@author: jarin.ritu
"""

import torch
from torch.optim import Optimizer

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, last_epoch=-1):
        self.max_iter = max_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter = self.last_epoch
        return [
            base_lr * (1 - current_iter / self.max_iter) ** 0.9
            for base_lr in self.base_lrs
        ]