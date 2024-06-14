import torch
import pdb
import torch.nn as nn
from torch.nn import functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        modules = []
        if mode == '2d':
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        else:
            modules.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
        if has_bn:
            modules.append(nn.BatchNorm2d(out_channels) if mode == '2d' else nn.BatchNorm3d(out_channels))
        if has_relu:
            modules.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.block(x)


class QCO_2d(nn.Module):
    def __init__(self, scale, level_num, num_classes=4, stride=1, dilation=1, out_chan= 144, epsilon=1e-6):
        super(QCO_2d, self).__init__()
        
        self.f1 = nn.Sequential(ConvBNReLU(3, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='2d'), nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='2d')
        self.out = nn.Sequential(ConvBNReLU(out_chan, 128, 1, 1, 0, has_bn=True, has_relu=True, mode='2d'), 
                                 ConvBNReLU(128, 128, 1, 1, 0, has_bn=True, has_relu=False, mode='2d'))
        self.scale = scale
        self.level_num = level_num
        conv_layer = self.f2.block[0]  # Accessing the first element of the Sequential module
        out_channels = conv_layer.out_channels 
        self.fc = nn.Linear(out_channels, num_classes) 
        self.epsilon = epsilon

    def forward(self, x):
        #pdb.set_trace()
        # sampler = AnchorBasedSampler() # 256 channels for 2nd conv layer of ResNet50
        # x = sampler(xx)
        N1, C1, H1, W1 = x.shape #[64, 16, 52, 47]
        if H1 // self.level_num != 0 or W1 // self.level_num != 0:
            x = F.adaptive_avg_pool2d(x, ((int(H1/self.level_num)*self.level_num), int(W1/self.level_num)*self.level_num))
        N, C, H, W = x.shape
        self.size_h = int(H / self.scale)
        self.size_w = int(W / self.scale)
        x_ave = F.adaptive_avg_pool2d(x, (self.scale, self.scale))
        x_ave_up = F.adaptive_avg_pool2d(x_ave, (H, W))
    
        cos_sim = (F.normalize(x_ave_up, dim=1) * F.normalize(x, dim=1)).sum(1)
        cos_sim = cos_sim.unsqueeze(1) 
        cos_sim = cos_sim.reshape(N, 1, self.scale, self.size_h, self.scale, self.size_w)
        cos_sim = cos_sim.permute(0, 1, 2, 4, 3, 5)
        cos_sim = cos_sim.reshape(N, 1, int(self.scale*self.scale), int(self.size_h*self.size_w)) 
        cos_sim = cos_sim.permute(0, 1, 3, 2) 
        cos_sim = cos_sim.squeeze(1)
        cos_sim_min, _ = cos_sim.min(1)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        

        q_levels = torch.arange(self.level_num).float().cuda()
        q_levels = q_levels.expand(N, self.scale*self.scale, self.level_num)
        q_levels =  (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min #Rescale q_levels to be in the range [cos_sim_min, cos_sim_max]
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] #Compute the quantization interval for each quantization level.
        q_levels_inter = q_levels_inter.unsqueeze(1).unsqueeze(-1) #(N, 1, self.scale*self.scale, 1)
        cos_sim = cos_sim.unsqueeze(-1) #(N, self.size_h*self.size_w, self.scale*self.scale, 1)
        q_levels = q_levels.unsqueeze(1) 
        sigma = 1/( self.level_num/2)
        
        quant = torch.exp(-((cos_sim.unsqueeze(-1).unsqueeze(1) - q_levels.unsqueeze(-1).unsqueeze(-1))**2) / (2 *sigma**2))
        quant = quant.view([N, self.size_h, self.size_w, self.scale*self.scale, self.level_num]) 
        quant = quant.permute(0, -2, -1, 1, 2) 
        quant = quant.view(N, -1, self.size_h, self.size_w)
        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        quant = quant.view(N, self.scale*self.scale, self.level_num, self.size_h+1, self.size_w+1)
        quant_left = quant[:, :, :, :self.size_h, :self.size_w].unsqueeze(3) 
        quant_right = quant[:, :, :, 1:, 1:].unsqueeze(2) 
        quant = quant_left * quant_right 
        quant = (quant - quant.min()) / (quant.max() - quant.min() + self.epsilon) 

        

        sta = quant.sum(-1).sum(-1)  
        sta = sta.unsqueeze(1)
        q_levels = q_levels.expand(self.level_num, N, 1, self.scale*self.scale, self.level_num)
        q_levels_h = q_levels.permute(1, 2, 3, 0, 4)
        q_levels_w = q_levels_h.permute(0, 1, 2, 4, 3)
        
        sta = torch.cat([q_levels_h, q_levels_w, sta], dim=1) #Counting C
        sta = sta.reshape(N, 3, self.scale * self.scale, -1)
        sta = self.f1(sta)
        sta = self.f2(sta)
        x_ave = x_ave.reshape(N, C, -1)
        x_ave = x_ave.expand(self.level_num*self.level_num, N, C, self.scale*self.scale)
        x_ave = x_ave.permute(1, 2, 3, 0)
        sta = torch.cat([x_ave, sta], dim=1)
        sta = self.out(sta)
        sta = sta.mean(-1)
        sta = sta.reshape(N, sta.shape[1], self.scale, self.scale)
        
        theta = 0.5  #
    
        # Apply denoising to the 'sta' tensor
        denoised_sta = self.denoise(sta, theta)
        denoised_sta = (denoised_sta - denoised_sta.min()) / (denoised_sta.max() - denoised_sta.min() + self.epsilon)

    
        # Flatten the denoised_sta tensor for classification
        flattened_sta = denoised_sta.view(denoised_sta.size(0), -1)
    

        return denoised_sta
    
    def denoise(self, sta, theta):
        # Calculate the maximum value in sta
        max_sta = torch.max(sta)
    
        # Calculate E_extra as per Equation 5
        E_extra = torch.sum(torch.max(sta - theta * max_sta, torch.tensor(0.0)))
    
        # Calculate the redistributed intensity per quantization level
        redistributed_intensity = E_extra / self.level_num
    
        # Apply intensity clipping and redistribution for each quantization level
        DE = torch.where(sta > theta * max_sta,
                         theta * max_sta + redistributed_intensity,
                         sta + redistributed_intensity)
    
        return DE
