import torch
from torch import nn
import torch.nn.functional as F

class CustomCDMLayer(nn.Module):
    def __init__(self, subband_level=2, num_classes=4, in_channels=1, epsilon=1e-6, device = None):  
        super(CustomCDMLayer, self).__init__()
        
        # Parameters
        self.subband_level = subband_level
        self.in_channels = in_channels
        self.directions = 2 ** self.subband_level
        self.epsilon = epsilon
        
        # Convolutional filters
        self.H = nn.Parameter(torch.ones(1, in_channels, 3, 3))
        self.G = nn.Parameter(torch.ones(1, in_channels, 3, 3))
        
        self.sobel_horizontal = nn.Parameter(torch.tensor([[-1, -2, -1],
                                                           [0, 0, 0],
                                                           [1, 2, 1]], dtype=torch.float32).unsqueeze(0).expand(in_channels, -1, -1, -1).transpose(0, 1))
        
        self.sobel_vertical = nn.Parameter(torch.tensor([[-1, 0, 1],
                                                         [-2, 0, 2],
                                                         [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).expand(in_channels, -1, -1, -1).transpose(0, 1))

    def forward(self, x):
       
        laplacian_pyramid = []
        for _ in range(self.subband_level):

            
            low_pass_result = F.conv2d(x, self.H, padding=(0, 2), stride=1)
            low_pass_result = low_pass_result.expand(-1, x.shape[1], -1, -1)
            low_pass_result_downsampled = F.avg_pool2d(low_pass_result, kernel_size=3, stride=1)
            low_pass_result_upsampled = F.interpolate(low_pass_result_downsampled, size=x.shape[-2:], mode='bilinear')
            
            laplacian = x - low_pass_result_upsampled
            laplacian_pyramid.append(laplacian.clone())  # Cloning
            x = low_pass_result_upsampled.clone()  # Cloning
    
        for level, laplacian in enumerate(laplacian_pyramid):
            level_dfb_subbands = []
            for angle in range(self.directions):
                if angle % 2 == 0:
                    directional_subband = F.conv2d(laplacian, self.sobel_vertical, padding=1)
                else:
                    directional_subband = F.conv2d(laplacian, self.sobel_horizontal, padding=1)
                
                min_val = directional_subband.min()
                max_val = directional_subband.max()
                if max_val - min_val < self.epsilon:
                    directional_subband = torch.zeros_like(directional_subband)
                else:
                    directional_subband = (directional_subband - min_val) / (max_val - min_val + self.epsilon)
                
                level_dfb_subbands.append(directional_subband.clone())  # Cloning
            
            combined_subband = torch.sum(torch.stack(level_dfb_subbands), dim=0)
            min_val = combined_subband.min()
            max_val = combined_subband.max()
            if max_val - min_val < self.epsilon:
                combined_subband = torch.zeros_like(combined_subband)
            else:
                combined_subband = (combined_subband - min_val) / (max_val - min_val + self.epsilon)
            
            reconstructed_subband = F.interpolate(combined_subband, size=laplacian.shape[-2:], mode='bilinear', align_corners=False)
            laplacian_pyramid[level] = laplacian + reconstructed_subband.clone()  # Cloning
    
        reconstructed_subbands = []
        for i in range(self.subband_level):
            expanded_G = self.G.expand(laplacian_pyramid[i].shape[1], -1, -1, -1)
            filtered_subband = F.conv2d(laplacian_pyramid[i], expanded_G, padding=2, stride=1)
            upsampled_subband = F.interpolate(filtered_subband, size=x.shape[-2:], mode='bilinear')
            
            min_val = upsampled_subband.min()
            max_val = upsampled_subband.max()
            if max_val - min_val < self.epsilon:
                upsampled_subband = torch.zeros_like(upsampled_subband)
            else:
                upsampled_subband = (upsampled_subband - min_val) / (max_val - min_val + self.epsilon)
            
            reconstructed_subbands.append(upsampled_subband.clone())  # Cloning
    
        H_min_size = min(subband.shape[2] for subband in reconstructed_subbands)
        W_min_size = min(subband.shape[3] for subband in reconstructed_subbands)
        reconstructed_subbands = [F.interpolate(subband, size=(H_min_size, W_min_size), mode='bilinear', align_corners=False) for subband in reconstructed_subbands]
        
        reconstructed_image = torch.cat(reconstructed_subbands, dim=1)
        min_val = reconstructed_image.min()
        max_val = reconstructed_image.max()
        if max_val - min_val < self.epsilon:
            reconstructed_image = torch.zeros_like(reconstructed_image)
        else:
            reconstructed_image = (reconstructed_image - min_val) / (max_val - min_val + self.epsilon)
    
        return reconstructed_image
