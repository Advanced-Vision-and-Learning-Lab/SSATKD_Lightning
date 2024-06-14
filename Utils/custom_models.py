import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import pdb
import lightning as L

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
class Simple_CNN(nn.Module):

    def __init__(self,in_channels,num_classes):
        super(Simple_CNN, self).__init__()
       
        self.in_channels = in_channels
        self.num_classes = num_classes

        #Define convolution layers
        self.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=(7,7),stride=2,
                               bias=True)
        self.conv2 = nn.Conv2d(16,32,kernel_size=(3,3),  bias=True)
        self.conv3 = nn.Conv2d(32,64,kernel_size=(3,3),  bias=True)
     
        #Define max pooling layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        
        #Fix size to make sure output is 6 x 6
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.fc = nn.Linear(self.conv3.out_channels*6*6,self.num_classes)
        

        self.relu = nn.ReLU(inplace=True)
        

    # Forward propogation
    def forward(self, x):
       
        #Pass through convolutionlayers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool1(x)
    
        #Pass through average pooling
        x = self.avgpool(x)
       
        #Pass through fully connected layer
        x = torch.flatten(x,start_dim=1)
       
        x = self.fc(x)
        return x
