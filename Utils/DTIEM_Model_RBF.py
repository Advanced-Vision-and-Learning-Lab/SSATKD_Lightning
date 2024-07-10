import torch
import pdb
import torch.nn as nn
from torch.nn import functional as F
# from point_features import point_sample



class QCO_2d(nn.Module):
    def __init__(self, scale, level_num, num_classes=4, stride=1, dilation=1, out_chan= 144, epsilon=1e-6):
        super(QCO_2d, self).__init__()       
        self.level_num = level_num
        self.scale = scale


    def forward(self, x):
        N1, C1, H1, W1 = x.shape #[64, 16, 52, 47]
        if H1 // self.level_num != 0 or W1 // self.level_num != 0:
            x = F.adaptive_avg_pool2d(x, ((int(H1/self.level_num)*self.level_num), int(W1/self.level_num)*self.level_num))
        N, C, H, W = x.shape
        self.size_h = int(H / self.scale)
        self.size_w = int(W / self.scale)
        x_ave = F.adaptive_avg_pool2d(x, (self.scale, self.scale)) #Global avg pooled feature Cx1x1
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
        # q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] #Compute the quantization interval for each quantization level.
        # q_levels_inter = q_levels_inter.unsqueeze(1).unsqueeze(-1) #(N, 1, self.scale*self.scale, 1)
        cos_sim = cos_sim.unsqueeze(-1) #(N, self.size_h*self.size_w, self.scale*self.scale, 1)
        q_levels = q_levels.unsqueeze(1) 
        sigma = 1/( self.level_num/2)
        # quant = torch.exp(-((cos_sim.unsqueeze(-1).unsqueeze(1) - q_levels.unsqueeze(-1).unsqueeze(-1))**2) * (2 *sigma**2))
        quant = torch.exp(-sigma**2 * ((cos_sim.unsqueeze(-1).unsqueeze(1) - q_levels.unsqueeze(-1).unsqueeze(-1))**2))

        quant = quant.view([N, self.size_h, self.size_w, self.scale*self.scale, self.level_num]) 
        quant = quant.permute(0, -2, -1, 1, 2) 
        quant = quant.view(N, -1, self.size_h, self.size_w)
        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        quant = quant.view(N, self.scale*self.scale,self.level_num, self.size_h+1, self.size_w+1)
        quant_left = quant[:, :, :, :self.size_h, :self.size_w].unsqueeze(3) 
        quant_right = quant[:, :, :, 1:, 1:].unsqueeze(2) 
        quant = quant_left * quant_right 
        # print(quant)

        # pdb.set_trace()
        sta = quant.sum(-1).sum(-1)  
        
        #Peform normalization (enforce sum to one constraint)
        sta = sta / sta.flatten(1).sum(-1)[:,None,None,None]
        sta = sta.unsqueeze(1)
        q_levels = q_levels.expand(self.level_num, N, 1, self.scale*self.scale, self.level_num)
        q_levels_h = q_levels.permute(1, 2, 3, 0, 4)
        q_levels_w = q_levels_h.permute(0, 1, 2, 4, 3)
        
        sta = torch.cat([q_levels_h, q_levels_w, sta], dim=1) #Counting C
        sta = sta.permute(0, 3, 4, 1, 2).squeeze(-1)
        # counting_numbers = sta[:, :, :, -1]
        # Flatten the counting numbers to (batch_size, bins)
        # counting_numbers = counting_numbers.view(N, -1)
        
        
        # sta = sta.reshape(N, 3, self.scale * self.scale, -1)
        return sta
    
