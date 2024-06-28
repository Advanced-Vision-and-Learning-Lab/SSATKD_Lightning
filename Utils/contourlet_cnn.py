import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_grayscale
from Utils.pycontourlet.pycontourlet4d.pycontourlet_module import Pycontourlet
import pdb
from Utils.pycontourlet.pycontourlet4d.dfilters import dfilters
from Utils.pycontourlet.pycontourlet4d.modulate2 import modulate2
from torch.nn import functional as F
from Utils.pycontourlet.pycontourlet4d.modulate2 import modulate2
from numpy import *
# try:
#     from Utils.pycontourlet.pycontourlet4d.resamp4c import resamp4c
# except:
from Utils.pycontourlet.pycontourlet4d.resamp4 import resamp4c
import matplotlib.pyplot as plt





# Ensure CUDA is available
print("CUDA Available: ", torch.cuda.is_available())
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def stack_same_dim(x):
    """Stack a list/dict of 4D tensors of the same img dimension together."""
    from collections import defaultdict
    
    output = defaultdict(list)
    
    if isinstance(x, list):
        for item in x:
            if isinstance(item, list):
                for tensor in item:
                    shape = tuple(tensor.shape)
                    output[shape].append(tensor)
            else:
                shape = tuple(item.shape)
                output[shape].append(item)
    else:
        for key, tensor in x.items():
            shape = tuple(tensor.shape[2:4])
            output[shape].append(tensor)
    
    for shape in output:
        output[shape] = torch.cat(output[shape], dim=1)
    
    return dict(output)

class ContourletCNN(nn.Module):
    """C-CNN: Contourlet Convolutional Neural Networks.
    
    Based on: 
    
    Parameters
    ----------
    n_levs : list of int, default = [0, 3, 3, 3]
         The numbers of DFB (Directional Filter Bank) decomposition levels at
         each pyramidal level from coarse to fine-scale.
         
         In each pyramidal level, there is one number of DFB decomposition
         levels is denoted as `l`, resulting in `2^l = 8` wedge-shaped
         subbands in the frequency domain. When the DFB decomposition level is
         0, resulting in 1 subband in the frequency domain, which is effectively
         a 2-D wavelet decomposition.
         
         For example:
             n_levs = [0, 3, 3, 3]
             num_subbands = [1+3, 2^3, 2^3, 2^3]
         
    variant : {"origin", "SSFF", "SSF"}, \
            default="SSF"
        The variants of the Contourlet-CNN model. From left to right, each
        variant is an incremental version of the previous variant, as such
        in an abalation study in the original paper.
        ``"origin"``:
            The 'origin' splices the elongated decomposed images into its
            corresponding sizes since the contourlet has elongated supports.
            No SSF features is concatenated to the features in FC2 layer.
        ``"SSFF"``:
            Instead of splicing, the 'SSFF' (spatial–spectral feature fusion)
            via contourlet directly resize the elongated decomposed images
            into its corresponding sizes. No SSF features is concatenated to
            the features in FC2 layer.
        ``"SSF"``:
            In addition to 'SSFF', the 'SFF' (statistical feature fusion)
            that denotes the additional texture features of decomposed images,
            are concatenated to the features in FC2 layer.
            The mean and variance of each subbands are chosen as the texture
            features of decomposed images.
        
    spec_type : {"avg", "all"}, \
            default="all"
            The type of spectral information to obtain from the image.
            ``'avg'``:
                The spectral information is obtained from the average value of
                all image channel.
            ``'all'``:
                The spectral information is obtained from each channels in the
                image.
    References
    ----------
    DOI : 10.1109/TNNLS.2020.3007412
    
    """
    def __init__(self, in_channels=16, n_levs=[0, 3, 3, 3], variant="SSF", spec_type="avg", pfilt="maxflat"):
        super(ContourletCNN, self).__init__()
        
        # Model hyperparameters
        self.in_channels = in_channels
        self.n_levs = n_levs
        self.variant = variant
        self.spec_type = spec_type
        self.device = torch.device("cuda:0")
        self.to()
        
        # Get the pyramidal filters from the filter name
        h, g = self.pfilters(pfilt)
        
        #Get row filtering weights (h0 and h1)
        h0, h1 = self.wfb2dec_filters(h,g,in_channels)
        
        #Get lowpass and high pass filters
        h_filter = self.sefilter2_filters(h,h,in_channels)
        g_filter = self.sefilter2_filters(g,g,in_channels)

        #Create convolution layers
        #Revisit group convolutions\
      

        self.row_filter_h0 = nn.Conv2d(h0.shape[0], in_channels, h0.shape[-2:], groups=in_channels,bias=False)  
        self.row_filter_h0.weight.data = h0
        self.row_filter_h0.weight.requires_grad = False
        
        self.row_filter_h1 = nn.Conv2d(h1.shape[0], in_channels, h1.shape[-2:], groups=in_channels,bias=False)  
        self.row_filter_h1.weight.data = h1
        self.row_filter_h1.weight.requires_grad = False
        
        self.se_filter_h_filter = nn.Conv2d(h_filter.shape[0], in_channels, h_filter.shape[-2:],groups=in_channels,bias=False)  
        self.se_filter_h_filter.weight.data = h_filter
        self.se_filter_h_filter.weight.requires_grad = False
        
        self.se_filter_g_filter = nn.Conv2d(g_filter.shape[0], in_channels, g_filter.shape[-2:],groups=in_channels,bias=False)  
        self.se_filter_g_filter.weight.data = g_filter
        self.se_filter_g_filter.weight.requires_grad = False
        

    
        
        #Create convolution layer for directional filter bank
    def __pdfbdec(self, x, method="resize"):
        # pdb.set_trace()
    
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i].cpu())
                img = to_grayscale(img)
                imgs.append(transforms.ToTensor()(img))
            # Restack and convert back to PyTorch tensor
            x = torch.stack(imgs, axis=0).to(self.device)

        coefs = self.batch_multi_channel_pdfbdec(x=x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3, 3])
        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(torch.max(torch.tensor([k[2], k[3]])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(torch.argmin(torch.tensor([k[2], k[3]]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        return coefs
    
    
    def forward(self, x):
        # pdb.set_trace()
        
        # Perform PDFB decomposition to obtain the coefficients and it's statistical features
        if self.variant == "origin":
            coefs = self.__pdfbdec(x, method="splice")
        else:
            coefs = self.__pdfbdec(x, method="resize")
        
        
        return coefs
    
    def to(self):
        self.device = torch.device("cuda:0")
        
    def batch_multi_channel_pdfbdec(self, x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3, 3]):
   
        if len(nlevs) == 0:
            y = [x]
        else:
            # Get the pyramidal filters from the filter name
            h, g = self.pfilters(pfilt)
            if nlevs[-1] != 0:
                # pdb.set_trace()
                # Laplacian decomposition
                xlo, xhi = self.lpdec(x, h, g)
                # DFB on the bandpass image
                if dfilt in ['pkva6', 'pkva8', 'pkva12', 'pkva']:
                    # Use the ladder structure (whihc is much more efficient)
                    xhi_dir = self.dfbdec_l(xhi, dfilt, nlevs[-1])
                else:
    
                    # General case
                    xhi_dir = self.dfbdec(xhi, dfilt, nlevs[-1])
    
            else:

                xlo, xLH, xHL, xHH = self.wfb2dec(x, h, g)
                xhi_dir = [xLH, xHL, xHH]
            # Recursive call on the low band
            ylo = self.batch_multi_channel_pdfbdec(xlo, pfilt, dfilt, nlevs[0:-1])
            y = ylo[:]
            y.append(xhi_dir)
            
        return y

    def dfbdec(self, x, fname, n):
        # pdb.set_trace()
        if (n != round(n)) or (n < 0):
            print('Number of decomposition levels must be a non-negative integer')
            return None
        
        if n == 0:
            return [x.clone()]
        
        # Get the diamond-shaped filters
        h0, h1 = dfilters(fname, 'd')
        k0 = modulate2(h0, 'c', None)
        k1 = modulate2(h1, 'c', None)
        
        def fbdec_level(x, k0, k1, levels):
            # pdb.set_trace()
            y = [None] * (2 ** levels)
            if levels == 1:
                y[0], y[1] = self.fbdec(x, k0, k1, 'q', '1r', 'per')
            else:
                x0, x1 = self.fbdec(x, k0, k1, 'q', '1r', 'per')
                y[0], y[1] = self.fbdec(x0, k0, k1, 'q', '2c', 'qper_col')
                y[2], y[3] = self.fbdec(x1, k0, k1, 'q', '2c', 'qper_col')
                f0, f1 = self.ffilters(h0, h1)
                
                for l in range(3, levels + 1):
                    y_old = y[:]
                    y = [None] * (2 ** l)
                    half_level = 2 ** (l - 2)
                    for k in range(half_level):
                        i = k % 2
                        y[2 * k], y[2 * k + 1] = self.fbdec(y_old[k], f0[i], f1[i], 'pq', i, 'per')
                    for k in range(half_level, 2 ** (l - 1)):
                        i = (k % 2) + 2
                        y[2 * k], y[2 * k + 1] = self.fbdec(y_old[k], f0[i], f1[i], 'pq', i, 'per')
            return y
        
        y = fbdec_level(x, k0, k1, n)
        y = self.backsamp(y)
        y[2 ** (n - 1):] = y[::-1][:2 ** (n - 1)]
        
        return y
    
    
    def pfilters(self, fname):
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    
        if fname in ["9/7", "9-7"]:
            h = torch.tensor([.037828455506995, -.023849465019380, -.11062440441842,
                              .37740285561265, .85269867900940, .37740285561265,
                              -.11062440441842, -.023849465019380, .037828455506995],
                             dtype=torch.float32)
            
            g = torch.tensor([-.064538882628938, -.040689417609558, .41809227322221,
                              .78848561640566, .41809227322221, -.040689417609558,
                              -.064538882628938], dtype=torch.float32)
            
        elif fname == "maxflat":
            M1 = M2 = 1 / sqrt2
            k1 = 1 - sqrt2
            k2 = k3 = M1
            
            h = torch.tensor([.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3], dtype=torch.float32) * M1
            h = torch.cat((h, h[:-1].flip(dims=[0])))
            
            g = torch.tensor([-.125 * k1 * k2 * k3, 0.25 * k1 * k2, -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                              1 + .5 * k1 * k2], dtype=torch.float32) * M2
            g = torch.cat((g, g[:-1].flip(dims=[0])))
            
            # Normalize
            h *= sqrt2
            g *= sqrt2
            
        elif fname in ["5/3", "5-3"]:
            h = torch.tensor([-1, 2, 6, 2, -1], dtype=torch.float32) / (4 * sqrt2)
            g = torch.tensor([1, 2, 1], dtype=torch.float32) / (2 * sqrt2)
            
        elif fname in ["burt", "Burt"]:
            h = torch.tensor([0.6, 0.25, -0.05], dtype=torch.float32)
            h = sqrt2 * torch.cat((h.flip(dims=[0]), h))
            
            g = torch.tensor([17.0 / 28, 73.0 / 280, -3.0 / 56, -3.0 / 280], dtype=torch.float32)
            g = sqrt2 * torch.cat((g.flip(dims=[0]), g))
            
        elif fname == "pkva":
            # Assuming ldfilter returns a PyTorch tensor
            beta = self.ldfilter(fname)
            
            lf = len(beta)
            n = lf // 2
            
            if lf % 2 != 0:
                raise ValueError("The input allpass filter must be even length")
            
            # beta(z^2)
            beta2 = torch.zeros(2 * lf - 1)
            beta2[::2] = beta
            
            # H(z)
            h = beta2.clone()
            h[2 * n - 1] += 1
            h /= 2
            
            # G(z)
            g = -torch.conv1d(beta2.unsqueeze(0).unsqueeze(0), h.unsqueeze(0).unsqueeze(0)).squeeze()
            g[4 * n - 2] += 1
            g[1:-1:2] = -g[1:-1:2]
            
            # Normalize
            h *= sqrt2
            g *= sqrt2
            
        else:
            raise ValueError("Filter name not recognized")
        
        return h, g
    
    
    def mod(self, a, b):
        return a % b
    
    def lpdec(self,x,h,g):
    
        # Lowpass filter and downsample
        xlo = self.sefilter2(x, h, h, 'per', None)
        c = xlo[:, :, ::2, ::2]
        
        # Compute the residual (bandpass) image by upsample, filter, and subtract
        # Even size filter needs to be adjusted to obtain perfect reconstruction
        adjust = self.mod(len(g) + 1, 2)
        xlo = torch.zeros(x.shape).to(x.device)
        xlo[:, :, ::2, ::2] = c
        tmp = self.sefilter2(xlo, g, g, 'per', adjust * array([1, 1]),h=False)
        d = x - tmp
    
        return c, d


    def wfb2dec_filters(self,h,g,in_channels):
        
       if torch.is_tensor(h):
           h = h.cpu().detach().numpy()
       if torch.is_tensor(g):
           g = g.cpu().detach().numpy()
    
       # Make sure filter in a row vector
       h = h[:, np.newaxis].reshape(len(h),)
       g = g[:, np.newaxis].reshape(len(g),)
    
       h0 = h
       len_h0 = len(h0)
       ext_h0 = floor(len_h0 / 2.0)
       # Highpass analysis filter: H1(z) = -z^(-1) G0(-z)
       len_h1 = len(g)
       c = floor((len_h1 + 1.0) / 2.0)
       # Shift the center of the filter by 1 if its length is even.
       if mod(len_h1, 2) == 0:
           c = c + 1
       # print(c)
       h1 = - g * (-1)**(np.arange(1, len_h1 + 1) - c)
       ext_h1 = len_h1 - c + 1   
       
       #Expand filters to correct shape for convolution
       h0 = torch.from_numpy(h0[:, np.newaxis].astype(np.float32))
       h0 = h0[None, None, :, :]
       h0 = h0.repeat(in_channels, 1, 1, 1)
       
       h1 = torch.from_numpy(h1[:, np.newaxis].astype(np.float32))
       h1 = h1[None, None, :, :]
       h1 = h1.repeat(in_channels, 1, 1, 1)
       
       return h0, h1
   
    def wfb2dec(self,x,h,g):
        
       if torch.is_tensor(h):
           h = h.cpu().detach().numpy()
       if torch.is_tensor(g):
           g = g.cpu().detach().numpy()
    
       # Make sure filter in a row vector
       h = h[:, np.newaxis].reshape(len(h),)
       g = g[:, np.newaxis].reshape(len(g),)
    
       h0 = h
       len_h0 = len(h0)
       ext_h0 = floor(len_h0 / 2.0)
       # Highpass analysis filter: H1(z) = -z^(-1) G0(-z)
       len_h1 = len(g)
       c = floor((len_h1 + 1.0) / 2.0)
       # Shift the center of the filter by 1 if its length is even.
       if mod(len_h1, 2) == 0:
           c = c + 1
       # print(c)
       h1 = - g * (-1)**(np.arange(1, len_h1 + 1) - c)
       ext_h1 = len_h1 - c + 1
       
       # Row-wise filtering
       x_L = self.rowfiltering(x, h0, ext_h0)
       x_L = x_L[:, :, :, ::2]  # (:, 1:2:end)
    
       x_H = self.rowfiltering(x, h1, ext_h1, h0=False)
       x_H = x_H[:, :, :, ::2]  # x_H(:, 1:2:end);
    
       # Column-wise filtering
       x_LL = self.rowfiltering(x_L.conj().permute(0, 1, 3, 2), h0, ext_h0)
       x_LL = x_LL.conj().permute(0, 1, 3, 2)
       x_LL = x_LL[:, :, ::2, :]
    
       x_LH = self.rowfiltering(x_L.conj().permute(0, 1, 3, 2), h1, ext_h1, h0 = False)
       x_LH = x_LH.conj().permute(0, 1, 3, 2)
       x_LH = x_LH[:, :, ::2, :]
    
       x_HL = self.rowfiltering(x_H.conj().permute(0, 1, 3, 2), h0, ext_h0)
       x_HL = x_HL.conj().permute(0, 1, 3, 2)
       x_HL = x_HL[:, :, ::2, :]
    
       x_HH = self.rowfiltering(x_H.conj().permute(0, 1, 3, 2), h1, ext_h1, h0 = False)
       x_HH = x_HH.conj().permute(0, 1, 3, 2)
       x_HH = x_HH[:, :, ::2, :]
    
       return x_LL, x_LH, x_HL, x_HH
    
    def rowfiltering(self, x, f, ext1, h0 = True):
        ext1 = int(ext1)
        ext2 = int(len(f) - ext1 - 1)
        x = torch.cat([x[:, :, :, -ext1:], x, x[:, :, :, :ext2]], dim=3)
        
        # Prepare inputs and filters for convolution
        inputs = x.conj().permute(0, 1, 3, 2)
        # filters = torch.from_numpy(f[:, np.newaxis].astype(np.float32))
        # filters = filters[None, None, :, :]
        # filters = filters.repeat(inputs.shape[1], 1, 1, 1)
        
        # Perform convolution
        if h0:
            y = self.row_filter_h0(inputs)
        else:
            y = self.row_filter_h1(inputs)
        # y = F.conv2d(inputs, filters, groups=inputs.size(1)).conj().permute(0, 1, 3, 2)
    
        return y
    


    def fbdec(self,x, h0, h1, type1, type2, extmod):
        # pdb.set_trace()

        if type1 == 'pq':
            x = self.resamp(x, type2, None, None)
    
        if all(mod(h1.shape, 2)):
            shift = array([[-1], [0]])

            if type1 == 'p':
                R = [[None]] * 4
                R[0] = array([[1, 1], [0, 1]])
                R[1] = array([[1, -1], [0, 1]])
                R[2] = array([[1, 0], [1, 1]])
                R[3] = array([[1, 0], [-1, 1]])
                shift = R[type2] * shift
        else:
            shift = array([[0], [0]])
        # Extend, filter and keep the original size
        y0 = self.efilter2(x, h0, extmod, None)
        y1 = self.efilter2(x, h1, extmod, shift)
        # Downsampling
        if type1 == 'q':
            # Quincunx downsampling
            y0 = self.qdown(y0, type2, None, None)
            y1 = self.qdown(y1, type2, None, None)
        elif type1 == 'p':
            # Parallelogram downsampling
            y0 = self.pdown(y0, type2, None, None)
            y1 = self.pdown(y1, type2, None, None)
        elif type1 == 'pq':
            # Quincux downsampling using the equipvalent type
            pqtype = ['1r', '2r', '2c', '1c']
            y0 = self.qdown(y0, pqtype[type2], None, None )
            y1 = self.qdown(y1, pqtype[type2], None, None)
        else:
            print("Invalid input type1")
    
        return y0, y1
    
    def ffilters(self,h0, h1):
        f0 = [[None]] * 4
        f1 = [[None]] * 4
    
        # For the first half channels
        f0[0] = modulate2(h0, 'r', None)
        f1[0] = modulate2(h1, 'r', None)
    
        f0[1] = modulate2(h0, 'c', None)
        f1[1] = modulate2(h1, 'c', None)
    
        # For the second half channels,
        # use the transposed filters of the first half channels
        f0[2] = f0[0].conj().T
        f1[2] = f1[0].conj().T
    
        f0[3] = f0[1].conj().T
        f1[3] = f1[1].conj().T
    
        return f0, f1
        
    def backsamp(self,y):
        # Number of decomposition tree levels
        n = int(log2(len(y)))
    
        if (n != round(n)) or (n < 1):
            print("Input must be a cell vector of dyadic length")
        if n == 1:
            # One level, the decomposition filterbank shoud be Q1r
            # Undo the last resampling (Q1r = R2 * D1 * R3)
            for k in range(0, 2):
                y[k] = self.resamp(y[k], 3, None, None)
                y[k][:, 0::2] = self.resamp(y[k][:, 0::2], 0, None, None)
                y[k][:, 1::2] = self.resamp(y[k][:, 1::2], 0, None, None)
    
        elif n > 2:
            N = 2**(n - 1)
            for k in range(0, 2**(n - 2)):
                shift = 2 * (k + 1) - (2**(n - 2) + 1)
                # The first half channels
                y[2 * k] = self.resamp(y[2 * k], 2, shift, None)
                y[2 * k + 1] = self.resamp(y[2 * k + 1], 2, shift, None)
                y[2 * k + N] = self.resamp(y[2 * k + N], 0, shift, None)
                y[2 * k + 1 + N] = self.resamp(y[2 * k + 1 + N], 0, shift, None)
    
        return y

    def sefilter2_filters(self, f1, f2, in_channels):

    
        # Ensure filters are on the correct device
        if not isinstance(f1, torch.Tensor):
            f1 = torch.tensor(f1, dtype=torch.float32, device=device)
        else:
            f1 = f1.clone().detach().to(device)
    
        if not isinstance(f2, torch.Tensor):
            f2 = torch.tensor(f2, dtype=torch.float32, device=device)
        else:
            f2 = f2.clone().detach().to(device)
    
        # Make sure filters are in the correct format
        f1 = f1.reshape(-1, 1)
        f2 = f2.reshape(-1, 1)
    
        filters = f1 @ f2.T
        filters = filters.unsqueeze(0).unsqueeze(0)
        filters = filters.repeat(in_channels, 1, 1, 1)
        
        return filters
   
    def sefilter2(self, x, f1, f2, extmod='per', shift=None, device = torch.device("cuda:0"),h=True):

        if extmod is None:
            extmod = 'per'
    
        if shift is None:
            shift = torch.tensor([0, 0], dtype=torch.float32, device=device)
        # Ensure filters are on the correct device
        if not isinstance(f1, torch.Tensor):
            f1 = torch.tensor(f1, dtype=torch.float32, device=device)
        else:
            f1 = f1.clone().detach().to(device)
    
        if not isinstance(f2, torch.Tensor):
            f2 = torch.tensor(f2, dtype=torch.float32, device=device)
        else:
            f2 = f2.clone().detach().to(device)
    
        # Make sure filters are in the correct format
        f1 = f1.reshape(-1, 1)
        f2 = f2.reshape(-1, 1)
        
        # Periodized extension
        lf1 = (len(f1) - 1) / 2.0
        lf2 = (len(f1) - 1) / 2.0

        y = self.extend2(x, floor(lf1) + shift[0], ceil(lf1) - shift[0],
                    floor(lf2) + shift[1], ceil(lf2) - shift[1], extmod)
    
        
        if h:
            y = self.se_filter_h_filter(y)

        else:
            y = self.se_filter_g_filter(y)
      
        
        return y

    
    def resamp(self, x, type_, shift, extmod):
        # pdb.set_trace()

        # if torch.is_tensor(x):
        #     x = x.cpu().detach().numpy()
        # else:
        #     x = x.astype(np.float32)
        
        if shift is None:
            shift = 1
    
        if extmod is None:
            extmod = 'per'
    
        if type_ == 0 or type_ == 1:
            # y = torch.from_numpy(resamp4c(x, type_, shift, extmod))
            y = resamp4c(x, type_, shift, extmod)
        elif type_ == 2 or type_ == 3:
            y = resamp4c(x.transpose(3, 2), type_ - 2, shift, extmod).transpose(3, 2)
        else:
            print("The second input (type_) must be one of {0, 1, 2, 3}")
    
        return y
    
    def efilter2(self, x, f, extmod='per', shift=None):
        if shift is None:
            shift = torch.tensor([[0], [0]])
        
        # Periodized extension
        if f.ndim < 2:
            sf = (torch.cat((torch.tensor([1]), torch.tensor(f.shape))) - 1) / 2.0
        else:
            sf = (torch.tensor(f.shape) - 1) / 2.0
        
        # Use PyTorch's floor and ceil functions
        ru = int(torch.floor(sf[0]) + shift[0][0])
        rd = int(torch.ceil(sf[0]) - shift[0][0])
        cl = int(torch.floor(sf[1]) + shift[1][0])
        cr = int(torch.ceil(sf[1]) - shift[1][0])
        
        # Continue with your operations
        xext = self.extend2(x, ru, rd, cl, cr, extmod)
    
        # Convolution and keep the central part that has the size as the input
        if f.ndim < 2:
            inputs = xext.float()  # Ensure input is float
            if not torch.is_tensor(f):
                filters = torch.from_numpy(f[:, np.newaxis].astype(np.float32))
            else:
                filters = f[:, np.newaxis].float()
            filters = filters[None, None, :, :]
            filters = torch.repeat_interleave(filters, x.shape[1], dim=0)
            y = torch.nn.functional.conv2d(inputs, filters, groups=inputs.size(1)).permute(0, 1, 3, 2)
        else:
            if not torch.is_tensor(xext):
                inputs = torch.from_numpy(xext.astype(np.float32))
            else:
                inputs = xext.float()  # Ensure input is float
            if not torch.is_tensor(f):
                filters = torch.from_numpy(f.astype(np.float32))
            else:
                filters = f.float()
            filters = filters[None, None, :, :]
            filters = torch.repeat_interleave(filters, x.shape[1], dim=0)
            try:
                y = torch.nn.functional.conv2d(inputs, filters, groups=inputs.size(1))
            except:
                y = torch.nn.functional.conv2d(inputs, filters.to(inputs.device), groups=inputs.size(1))
    
        return y

    
    
    def qdown(self, x, type, extmod, phase):
        # pdb.set_trace()
        if type is None:
            type = '1r'
    
        if phase is None:
            phase = 0
    
        if type == '1r':
            z = self.resamp(x, 1, None, None)
            if phase == 0:
                y = self.resamp(z[:, :, ::2, :], 2, None, None)
            else:
                y = self.resamp(hstack((z[:, :, 1::2, 1:], z[:, :, 1::2, 0:1])), 2, None, None)
    
        elif type == '1c':
            z = self.resamp(x, 2, None, None)
            if phase == 0:
                y = self.resamp(z[:, :, :, ::2], 1, None, None)
            else:
                y = self.resamp(z[:, :, :, 1::2], 1, None, None)
        elif type == '2r':
            z = self.resamp(x, 0, None, None)
            if phase == 0:
                y = self.resamp(z[:, :, ::2, :], 3, None, None)
            else:
                y = self.resamp(z[:, :, 1::2, :], 3, None, None)
        elif type == '2c':
            z = self.resamp(x, 3, None, None)
            if phase == 0:
                y = self.resamp(z[:, :, :, ::2], 0, None, None)
            else:
                y = self.resamp(hstack((z[:, :, 1:, 1::2].conj().transpose(0, 1, 3, 2),
                                   z[:, :, 0:1, 1::2].conj().transpose(0, 1, 3, 2))).conj().transpose(0, 1, 3, 2), 
                           0, None, None)
        else:
            print("Invalid argument type")
        return y
    
    
    def extend2(self, x, ru, rd, cl, cr, extmod):
        _, _, rx, cx = x.shape
    
        if extmod == 'per':
            I = self.getPerIndices(rx, ru, rd)
            y = x[:, :, I, :]
    
            I = self.getPerIndices(cx, cl, cr)
            y = y[:, :, :, I]
    
            return y
    
        elif extmod == 'qper_row':
            rx2 = round(rx / 2.0)
            y = torch.cat([torch.cat([x[:, :, rx2:rx, cx - cl:cx], x[:, :, 0:rx2, cx - cl:cx]], dim=2), 
                           x, 
                           torch.cat([x[:, :, rx2:rx, 0:cr], x[:, :, 0:rx2, 0:cr]], dim=2)], dim=3)
            I = self.getPerIndices(rx, ru, rd)
            y = y[:, :, I, :]
            return y
    
        elif extmod == 'qper_col':
            cx2 = int(round(cx / 2.0))
            y = torch.cat([
                torch.cat([x[:, :, rx - ru:rx, cx2:cx], x[:, :, rx - ru:rx, 0:cx2]], dim=3),
                x,
                torch.cat([x[:, :, 0:rd, cx2:cx], x[:, :, 0:rd, 0:cx2]], dim=3)
            ], dim=2)
    
            I = self.getPerIndices(cx, cl, cr)
            y = y[:, :, :, I]
            return y
    
        else:
            raise ValueError("Invalid input for EXTMOD")

    def getPerIndices(self, lx, lb, le):

        # Create ranges directly on GPU
        I1 = torch.arange(lx - lb, lx)
        I2 = torch.arange(0, lx)
        I3 = torch.arange(0, le)
        
    
        # Concatenate tensors on GPU
        I = torch.cat((I1, I2, I3))
    
    
        # Wrap around the indices if they are out of bounds
        if (lx < lb) or (lx < le):
            I = torch.remainder(I, lx)
            I[I == 0] = lx
    
    
        # Clamp operation on GPU
        I = torch.clamp(I, 0, lx - 1)
    
        return I.to(torch.int)
