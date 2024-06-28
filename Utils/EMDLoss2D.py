# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:41:48 2024

@author: jpeeples
"""

import torch
import torch.nn as nn
import pdb

class EMDLoss2D(nn.Module):
    def __init__(self):
        super(EMDLoss2D, self).__init__()

    def forward(self, pred_hist, target_hist, bin_centers_x, bin_centers_y):

        # Ensure the histograms are normalized
        pred_hist = pred_hist / (torch.sum(pred_hist, dim=(-1, -2), keepdim=True) + 1e-6)
        target_hist = target_hist / (torch.sum(target_hist, dim=(-1, -2), keepdim=True) + 1e-6)

        # Calculate the cumulative distribution functions (CDFs)
        cdf_pred = torch.cumsum(torch.cumsum(pred_hist, dim=-1), dim=-2)
        cdf_target = torch.cumsum(torch.cumsum(target_hist, dim=-1), dim=-2)

        # Calculate the distances between each pair of 2D bin centers
        bin_centers_x_diff = bin_centers_x.unsqueeze(-1) - bin_centers_x.unsqueeze(0)
        bin_centers_y_diff = bin_centers_y.unsqueeze(-1) - bin_centers_y.unsqueeze(0)
        bin_diffs = torch.sqrt(bin_centers_x_diff**2 + bin_centers_y_diff**2)

        # Expand bin_diffs to match the dimensions of cdf_pred and cdf_target
        bin_diffs = bin_diffs.unsqueeze(0).unsqueeze(0)

        # Calculate the EMD by summing the weighted absolute differences between the CDFs
        emd = torch.sum(bin_diffs * torch.abs(cdf_pred - cdf_target), dim=(-1, -2))

        return torch.mean(emd)

# Example usage
if __name__ == '__main__':
    # Create random 2D histograms and bin centers for the example
    batch_size = 4
    num_bins_x = 5  # Number of bins in the x dimension
    num_bins_y = 5  # Number of bins in the y dimension
    pred_hist = torch.rand(batch_size, num_bins_x, num_bins_y)
    target_hist = torch.rand(batch_size, num_bins_x, num_bins_y)
    bin_centers_x = torch.linspace(0, 1, num_bins_x)  # Example bin centers for x dimension
    bin_centers_y = torch.linspace(0, 1, num_bins_y)  # Example bin centers for y dimension

    # Create the EMD loss function
    emd_loss = EMDLoss2D()

    # Compute the EMD loss between the predicted and target 2D histograms
    loss = emd_loss(pred_hist, target_hist, bin_centers_x, bin_centers_y)
    
    print(loss)  # Should print the EMD loss value