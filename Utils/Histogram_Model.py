## PyTorch dependencies
import torch.nn as nn
import numpy as np
import torch
from Utils.TDNN_Model import TDNN

class HistRes(nn.Module):
    
    def __init__(self,histogram_layer, parallel=True,model_name ='resnet18',
                 add_bn=True,scale=5,pretrained=True, TDNN_feats = 1,subband_level=2,device=None,num_class = 4,):
        
        #inherit nn.module
        super(HistRes,self).__init__()
        self.parallel = parallel
        self.add_bn = add_bn
        self.scale = scale
        self.model_name = model_name
        self.bn_norm = None
        self.fc = None
        self.dropout = None
        
        #Default to use resnet18, otherwise use Resnet50
        #Defines feature extraction backbone model and redefines linear layer        
        if model_name == "TDNN":
            self.backbone = TDNN(in_channels=TDNN_feats)
            num_ftrs = self.backbone.fc.in_features
            self.dropout = self.backbone.dropout

        else: 
            print('Model not defined')
            
        if self.add_bn:
            if self.bn_norm is None:
                self.bn_norm = nn.BatchNorm2d(num_ftrs)
            else:
                pass
        
        if self.dropout is None:
            self.dropout = nn.Sequential()
            
        
        #Define histogram layer and fc
        self.histogram_layer = histogram_layer
        
        #Change histogram layer pooling 
        output_size = int(num_ftrs / histogram_layer.bin_widths_conv.out_channels)
        histogram_layer.hist_pool = nn.AdaptiveAvgPool2d(int(np.sqrt(output_size)))
        
        if self.fc is None:
            self.fc = self.backbone.fc
            self.backbone.fc = torch.nn.Sequential()
        
        
    def forward(self,x):

        #Only use histogram features at end of network       
        if self.model_name == 'TDNN':
            x = self.backbone.conv1(x)
            x = self.backbone.nonlinearity(x)
            x = self.backbone.maxpool1(x)
            
            x = self.backbone.conv2(x)
            x = self.backbone.nonlinearity(x)
            x2 = self.backbone.maxpool2(x)
            # x_cdm = self.custom_cdm_layer(x)
            # x_dtiem = self.custom_ditm_layer(x)
            
            x = self.backbone.conv3(x2)
            x = self.backbone.nonlinearity(x)
            x = self.backbone.maxpool3(x)
            
            x = self.backbone.conv4(x)
            x = self.backbone.nonlinearity(x)
            x = self.backbone.maxpool4(x)
        
        #All ResNet models
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if(self.parallel):
            if self.add_bn:
                if self.model_name == 'TDNN':
                    x_pool = torch.flatten(x,start_dim=-2)
                    x_pool = self.backbone.conv5(x_pool)
                    x_pool = self.backbone.sigmoid(x_pool)
                    x_pool = self.backbone.avgpool(x_pool)
                    x_pool = torch.flatten(self.bn_norm(x_pool.unsqueeze(-1)),start_dim=1)
                   
                else:
                    x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
            else:
                if self.model_name == 'TDNN':
                    x_pool = torch.flatten(x,start_dim=-2)
                    x_pool = self.backbone.conv5(x_pool)
                    x_pool = self.backbone.sigmoid(x_pool)
                    x_pool = self.backbone.avgpool(x_pool)
                    x_pool = torch.flatten(x_pool,start_dim=1)
                else:
                    x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
            # pdb.set_trace()
  
            x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
            x_combine = torch.cat((x_pool,x_hist),dim=1)
            x_combine = self.dropout(x_combine)
            output = self.fc(x_combine)
        else:
            x = torch.flatten(self.histogram_layer(x),start_dim=1)
            x = self.dropout(x)
            output = self.fc(x)
     
        return x2, output
    
        
        
        
        