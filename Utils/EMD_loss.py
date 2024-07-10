
from torch import nn
import torch
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class EarthMoversDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # pdb.set_trace()
        
        # # input has dims: (Batch x Bins)
        # bins = x.size(1)
        # r = torch.arange(bins)
        # s, t = torch.meshgrid(r, r)
        # tt = (t >= s).float().to(device)
        
        cdf_student = torch.cumsum(x,dim=1)
        cdf_teacher = torch.cumsum(y,dim=1)

        # cdf_x = torch.matmul(x, tt.float())
        # cdf_y = torch.matmul(y, tt.float())

    
        return torch.sum(((cdf_student - cdf_teacher)**2),axis=1).mean()


class MutualInformationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p1, p2):
        
        # input p12 has dims: (Batch x Bins x Bins)
        # input p1 & p2 has dims: (Batch x Bins)
        p12 = torch.bmm(p1.unsqueeze(2),p2.unsqueeze(1))
        
        #normalize
        p12 = p12 / p12.flatten(1).sum(-1)[:,None,None]
        
        #Compute joint distribution
        product_p = torch.matmul(torch.transpose(p1.unsqueeze(1), 1, 2), p2.unsqueeze(1)) + torch.finfo(p1.dtype).eps
        mi = torch.sum(p12 * torch.log(p12 / product_p + torch.finfo(p1.dtype).eps), dim=(1, 2))
        h = -torch.sum(p12 * torch.log(p12 + torch.finfo(p1.dtype).eps), dim=(1, 2))

        return (1 - (mi / h)).mean()
    


class EMDLoss2D(nn.Module):
    def __init__(self):
        super(EMDLoss2D, self).__init__()

    def forward(self, student_hist, teacher_hist):
        # pdb.set_trace()
        # Extract the counts from the last dimension of the histograms
        pred_hist_counts = student_hist[:, :, :, -1]
        target_hist_counts = teacher_hist[:, :, :, -1]

        # Extract bin_x and bin_y from the feature maps
        pred_bin_x = student_hist[:, :, :, 0]
        pred_bin_y = student_hist[:, :, :, 1]
        target_bin_x = teacher_hist[:, :, :, 0]
        target_bin_y = teacher_hist[:, :, :, 1]
        
        # print("\npred_bin_x",pred_bin_x[:2,:2,:2])
        # print("\npred_bin_y",pred_bin_y[:2,:2,:2])
        
        # pdb.set_trace()

        # Ensure the histograms are normalized
        pred_hist_counts = pred_hist_counts / (torch.sum(pred_hist_counts, dim=(-1, -2), keepdim=True) + 1e-6)
        target_hist_counts = target_hist_counts / (torch.sum(target_hist_counts, dim=(-1, -2), keepdim=True) + 1e-6)

        # Calculate the cumulative distribution functions (CDFs)
        cdf_pred = torch.cumsum(torch.cumsum(pred_hist_counts, dim=-1), dim=-2)
        cdf_target = torch.cumsum(torch.cumsum(target_hist_counts, dim=-1), dim=-2)

        # Calculate the distances between each pair of 2D bin centers
        bin_centers_x_diff = pred_bin_x.unsqueeze(-1) - target_bin_x.unsqueeze(-3)
        bin_centers_y_diff = pred_bin_y.unsqueeze(-1) - target_bin_y.unsqueeze(-3)
        # print("\nbin_centers_x_diff",bin_centers_x_diff[:2,:2,:2,:2])
        # print("\nbin_centers_y_diff",bin_centers_y_diff[:2,:2,:2,:2])
        ground_distance = torch.sqrt(bin_centers_x_diff**2 + bin_centers_y_diff**2)
        # print("\nground_distance",ground_distance[:2,:2,:2,:2])

        # Expand bin_diffs to match the dimensions of cdf_pred and cdf_target
        ground_distance = ground_distance.unsqueeze(0).unsqueeze(0)

        # Calculate the EMD by summing the weighted absolute differences between the CDFs
        emd = torch.sum(ground_distance * torch.abs(cdf_pred.unsqueeze(-1) - cdf_target.unsqueeze(-3)), dim=(-1, -2))

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