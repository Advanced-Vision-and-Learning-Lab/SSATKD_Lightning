#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:23:44 2024

@author: jarin.ritu
"""
from numpy import *
import numpy as np
import torch
import torch.nn as nn
from .dfilters import dfilters
from .modulate2 import modulate2
from torch.nn import functional as F
from .modulate2 import modulate2
try:
    from .resamp4c import resamp4c
except:
    from .resamp4 import resamp4c

import pdb


class Pycontourlet(nn.Module):
    def __int__(self, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3, 3],device=torch.device("cuda:0")):
        super(Pycontourlet, self).__init__()
        
        
        self.nlevs = nlevs
        self.pfilt=pfilt
        self.dfilt=dfilt
        self.device = device
        
    
    
    def batch_multi_channel_pdfbdec(self, x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3, 3],device = torch.device("cuda:0")):
        
        if len(nlevs) == 0:
            y = [x]
        else:
            # Get the pyramidal filters from the filter name
            h, g = self.pfilters(pfilt, device=device)
            if nlevs[-1] != 0:
                # Laplacian decomposition
                xlo, xhi = self.lpdec(x, h, g, device=device)
                # DFB on the bandpass image
                if dfilt in ['pkva6', 'pkva8', 'pkva12', 'pkva']:
                    # Use the ladder structure (whihc is much more efficient)
                    xhi_dir = self.dfbdec_l(xhi, dfilt, nlevs[-1],  device=device)
                else:
    
                    # General case
                    xhi_dir = self.dfbdec(xhi, dfilt, nlevs[-1],  device=device)
    
            else:

                xlo, xLH, xHL, xHH = self.wfb2dec(x, h, g,  device=device)
                xhi_dir = [xLH, xHL, xHH]
            # Recursive call on the low band
            ylo = self.batch_multi_channel_pdfbdec(xlo, pfilt, dfilt, nlevs[0:-1],  device=device)
            y = ylo[:]
            y.append(xhi_dir)
            
        return y

    def dfbdec(self,x, fname, n, device = torch.device("cuda:0")):
        if (n != round(n)) or (n < 0):
            print('Number of decomposition levels must be a non-negative integer')
    
        if n == 0:
            return [x.clone().to(device)]
    
        # Get the diamond-shaped filters
        h0, h1 = dfilters(fname, 'd')
        k0 = modulate2(h0, 'c', None)
        k1 = modulate2(h1, 'c', None)
        # Tree-structured filter banks
        if n == 1:
            # Simplest case, one level
            y = [None, None]
            y[0], y[1] = self.fbdec(x, k0, k1, 'q', '1r', 'per', device=device)
        else:
            # For the cases that n >= 2
            # First level
            x0, x1 = self.fbdec(x, k0, k1, 'q', '1r', 'per', device=device)
            # Second level
            y = [[None]] * 4
            y[0], y[1] = self.fbdec(x0, k0, k1, 'q', '2c', 'qper_col', device=device)
            y[2], y[3] = self.fbdec(x1, k0, k1, 'q', '2c', 'qper_col', device=device)
            # Fan filters from diamond filters
            f0, f1 = self.ffilters(h0, h1, device=device)
            # Now expand the rest of the tree
            for l in range(3, n + 1):
                # Allocate space for the new subband outputs
                y_old = y[:]
                y = [[None]] * 2**l
                # The first half channels use R1 and R2
                for k in range(0, 2**(l - 2)):
                    i = k % 2
                    y[2 * k], y[2 * k + 1] = self.fbdec(y_old[k],
                                                   f0[i], f1[i], 'pq', i, 'per', device=device)
                # The second half channels use R3 and R4
                for k in range(2**(l - 2), 2**(l - 1)):
                    i = (k % 2) + 2
                    y[2 * k], y[2 * k + 1] = self.fbdec(y_old[k],
                                                   f0[i], f1[i], 'pq', i, 'per', device=device)

        y = self.backsamp(y, device=device)
        y[2**(n - 1)::] = y[::-1][:2**(n - 1)]

        return y
    
    
    def pfilters(self, fname, device = torch.device("cuda:0")):
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=device))
    
        if fname in ["9/7", "9-7"]:
            h = torch.tensor([.037828455506995, -.023849465019380, -.11062440441842,
                              .37740285561265, .85269867900940, .37740285561265,
                              -.11062440441842, -.023849465019380, .037828455506995],
                             dtype=torch.float32, device=device)
            
            g = torch.tensor([-.064538882628938, -.040689417609558, .41809227322221,
                              .78848561640566, .41809227322221, -.040689417609558,
                              -.064538882628938], dtype=torch.float32, device=device)
            
        elif fname == "maxflat":
            M1 = M2 = 1 / sqrt2
            k1 = 1 - sqrt2
            k2 = k3 = M1
            
            h = torch.tensor([.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3], dtype=torch.float32, device=device) * M1
            h = torch.cat((h, h[:-1].flip(dims=[0])))
            
            g = torch.tensor([-.125 * k1 * k2 * k3, 0.25 * k1 * k2, -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                              1 + .5 * k1 * k2], dtype=torch.float32, device=device) * M2
            g = torch.cat((g, g[:-1].flip(dims=[0])))
            
            # Normalize
            h *= sqrt2
            g *= sqrt2
            
        elif fname in ["5/3", "5-3"]:
            h = torch.tensor([-1, 2, 6, 2, -1], dtype=torch.float32, device=device) / (4 * sqrt2)
            g = torch.tensor([1, 2, 1], dtype=torch.float32, device=device) / (2 * sqrt2)
            
        elif fname in ["burt", "Burt"]:
            h = torch.tensor([0.6, 0.25, -0.05], dtype=torch.float32, device=device)
            h = sqrt2 * torch.cat((h.flip(dims=[0]), h))
            
            g = torch.tensor([17.0 / 28, 73.0 / 280, -3.0 / 56, -3.0 / 280], dtype=torch.float32, device=device)
            g = sqrt2 * torch.cat((g.flip(dims=[0]), g))
            
        elif fname == "pkva":
            # Assuming ldfilter returns a PyTorch tensor
            beta = self.ldfilter(fname).to(device)
            
            lf = len(beta)
            n = lf // 2
            
            if lf % 2 != 0:
                raise ValueError("The input allpass filter must be even length")
            
            # beta(z^2)
            beta2 = torch.zeros(2 * lf - 1, device=device)
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
    
    def lpdec(self,x,h,g,device = torch.device("cuda:0")):
    
        # Lowpass filter and downsample
        xlo = self.sefilter2(x, h, h, 'per', None, device=device)
        c = xlo[:, :, ::2, ::2]
        
        # Compute the residual (bandpass) image by upsample, filter, and subtract
        # Even size filter needs to be adjusted to obtain perfect reconstruction
        adjust = mod(len(g) + 1, 2)
        xlo = torch.zeros(x.shape).to(device)
        xlo[:, :, ::2, ::2] = c
        tmp = self.sefilter2(xlo, g, g, 'per', adjust * array([1, 1]), device=device)
        d = x.to(device) - tmp.to(device)
    
        return c, d

    
    def wfb2dec(self,x,h,g,device = torch.device("cuda:0")):
        
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
       
       # pdb.set_trace()
    
       # Row-wise filtering
       x_L = self.rowfiltering(x, h0, ext_h0, device=device)
       x_L = x_L[:, :, :, ::2]  # (:, 1:2:end)
    
       x_H = self.rowfiltering(x, h1, ext_h1, device=device)
       x_H = x_H[:, :, :, ::2]  # x_H(:, 1:2:end);
    
       # Column-wise filtering
       x_LL = self.rowfiltering(x_L.conj().permute(0, 1, 3, 2), h0, ext_h0, device=device)
       x_LL = x_LL.conj().permute(0, 1, 3, 2)
       x_LL = x_LL[:, :, ::2, :]
    
       x_LH = self.rowfiltering(x_L.conj().permute(0, 1, 3, 2), h1, ext_h1, device=device)
       x_LH = x_LH.conj().permute(0, 1, 3, 2)
       x_LH = x_LH[:, :, ::2, :]
    
       x_HL = self.rowfiltering(x_H.conj().permute(0, 1, 3, 2), h0, ext_h0, device=device)
       x_HL = x_HL.conj().permute(0, 1, 3, 2)
       x_HL = x_HL[:, :, ::2, :]
    
       x_HH = self.rowfiltering(x_H.conj().permute(0, 1, 3, 2), h1, ext_h1, device=device)
       x_HH = x_HH.conj().permute(0, 1, 3, 2)
       x_HH = x_HH[:, :, ::2, :]
    
       return x_LL, x_LH, x_HL, x_HH
    
    def rowfiltering(self, x, f, ext1,device = torch.device("cuda:0")):
        ext1 = int(ext1)
        ext2 = int(len(f) - ext1 - 1)
        x = torch.cat([x[:, :, :, -ext1:], x, x[:, :, :, :ext2]], dim=3)
        
        # Prepare inputs and filters for convolution
        inputs = x.conj().permute(0, 1, 3, 2).to(device)
        filters = torch.from_numpy(f[:, np.newaxis].astype(np.float32)).to(device)
        filters = filters[None, None, :, :]
        filters = filters.repeat(inputs.shape[1], 1, 1, 1)
        
        # Perform convolution
        y = F.conv2d(inputs, filters, groups=inputs.size(1)).conj().permute(0, 1, 3, 2)
    
        return y
    
    def mod(self, a, b):
        return a % b


    def fbdec(self,x, h0, h1, type1, type2, extmod,device = torch.device("cuda:0")):

        if type1 == 'pq':
            x = self.resamp(x, type2, None, None, device=device)
    
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
        y0 = self.efilter2(x, h0, extmod, None, device=device)
        y1 = self.efilter2(x, h1, extmod, shift, device=device)
        # Downsampling
        if type1 == 'q':
            # Quincunx downsampling
            y0 = self.qdown(y0, type2, None, None, device=device)
            y1 = self.qdown(y1, type2, None, None, device=device)
        elif type1 == 'p':
            # Parallelogram downsampling
            y0 = self.pdown(y0, type2, None, None, device=device)
            y1 = self.pdown(y1, type2, None, None, device=device)
        elif type1 == 'pq':
            # Quincux downsampling using the equipvalent type
            pqtype = ['1r', '2r', '2c', '1c']
            y0 = self.qdown(y0, pqtype[type2], None, None,device=device )
            y1 = self.qdown(y1, pqtype[type2], None, None, device=device)
        else:
            print("Invalid input type1")
    
        return y0, y1
    
    def ffilters(self,h0, h1,device = torch.device("cuda:0")):
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
        
    def backsamp(self,y,device = torch.device("cuda:0")):
        # Number of decomposition tree levels
        n = int(log2(len(y)))
    
        if (n != round(n)) or (n < 1):
            print("Input must be a cell vector of dyadic length")
        if n == 1:
            # One level, the decomposition filterbank shoud be Q1r
            # Undo the last resampling (Q1r = R2 * D1 * R3)
            for k in range(0, 2):
                y[k] = self.resamp(y[k], 3, None, None, device=device)
                y[k][:, 0::2] = self.resamp(y[k][:, 0::2], 0, None, None, device=device)
                y[k][:, 1::2] = self.resamp(y[k][:, 1::2], 0, None, None, device=device)
    
        elif n > 2:
            N = 2**(n - 1)
            for k in range(0, 2**(n - 2)):
                shift = 2 * (k + 1) - (2**(n - 2) + 1)
                # The first half channels
                y[2 * k] = self.resamp(y[2 * k], 2, shift, None, device=device)
                y[2 * k + 1] = self.resamp(y[2 * k + 1], 2, shift, None, device=device)
                # The second half channels
                y[2 * k + N] = self.resamp(y[2 * k + N], 0, shift, None, device=device)
                y[2 * k + 1 + N] = self.resamp(y[2 * k + 1 + N], 0, shift, None, device=device)
    
        return y

    def sefilter2(self, x, f1, f2, extmod='per', shift=None, device = torch.device("cuda:0")):

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
                    floor(lf2) + shift[1], ceil(lf2) - shift[1], extmod, device=device)
    
        filters = f1 @ f2.T
        filters = filters.unsqueeze(0).unsqueeze(0)
        filters = filters.repeat(y.shape[1], 1, 1, 1)
        
        y = F.conv2d(y, filters, groups=y.size(1))
        
        return y

    
    def resamp(self, x, type_, shift, extmod,device = torch.device("cuda:0")):

        if torch.is_tensor(x):
            x = x.cpu().detach().numpy()
        else:
            x = x.astype(np.float32)
        
        if shift is None:
            shift = 1
    
        if extmod is None:
            extmod = 'per'
    
        if type_ == 0 or type_ == 1:
            y = torch.from_numpy(resamp4c(x, type_, shift, extmod)).to(device)
        elif type_ == 2 or type_ == 3:
            y = torch.from_numpy(resamp4c(x.transpose(0, 1, 3, 2), type_ - 2, shift, extmod).transpose(0, 1, 3, 2)).to(device)
        else:
            print("The second input (type_) must be one of {0, 1, 2, 3}")
    
        return y
    
    def efilter2(self, x, f, extmod='per', shift=None,device = torch.device("cuda:0")):
        if shift is None:
            shift = torch.tensor([[0], [0]], device=device)
        
        # Periodized extension
        if f.ndim < 2:
            sf = (torch.cat((torch.tensor([1], device=device), torch.tensor(f.shape, device=device))) - 1) / 2.0
        else:
            sf = (torch.tensor(f.shape, device=device) - 1) / 2.0
        
        # Use PyTorch's floor and ceil functions
        ru = int(torch.floor(sf[0]) + shift[0][0])
        rd = int(torch.ceil(sf[0]) - shift[0][0])
        cl = int(torch.floor(sf[1]) + shift[1][0])
        cr = int(torch.ceil(sf[1]) - shift[1][0])
        
        # Continue with your operations
        xext = self.extend2(x, ru, rd, cl, cr, extmod, device)
    
        # Convolution and keep the central part that has the size as the input
        if f.ndim < 2:
            inputs = xext.to(device).float()  # Ensure input is float
            if not torch.is_tensor(f):
                filters = torch.from_numpy(f[:, np.newaxis].astype(np.float32)).to(device)
            else:
                filters = f[:, np.newaxis].float().to(device)  # Ensure filter is float
            filters = filters[None, None, :, :]
            filters = torch.repeat_interleave(filters, x.shape[1], dim=0)
            y = torch.nn.functional.conv2d(inputs, filters, groups=inputs.size(1)).permute(0, 1, 3, 2)
        else:
            if not torch.is_tensor(xext):
                inputs = torch.from_numpy(xext.astype(np.float32)).to(device)
            else:
                inputs = xext.to(device).float()  # Ensure input is float
            if not torch.is_tensor(f):
                filters = torch.from_numpy(f.astype(np.float32)).to(device)
            else:
                filters = f.float().to(device)  # Ensure filter is float
            filters = filters[None, None, :, :]
            filters = torch.repeat_interleave(filters, x.shape[1], dim=0)
            y = torch.nn.functional.conv2d(inputs, filters, groups=inputs.size(1))
    
        return y

    
    
    def qdown(self, x, type, extmod, phase, device = torch.device("cuda:0")):
        if type is None:
            type = '1r'
    
        if phase is None:
            phase = 0
    
        if type == '1r':
            z = self.resamp(x, 1, None, None, device=device)
            if phase == 0:
                y = self.resamp(z[:, :, ::2, :], 2, None, None, device=device)
            else:
                y = self.resamp(hstack((z[:, :, 1::2, 1:], z[:, :, 1::2, 0:1])), 2, None, None, device=device)
    
        elif type == '1c':
            z = self.resamp(x, 2, None, None, device=device)
            if phase == 0:
                y = self.resamp(z[:, :, :, ::2], 1, None, None, device=device)
            else:
                y = self.resamp(z[:, :, :, 1::2], 1, None, None, device=device)
        elif type == '2r':
            z = self.resamp(x, 0, None, None, device=device)
            if phase == 0:
                y = self.resamp(z[:, :, ::2, :], 3, None, None, device=device)
            else:
                y = self.resamp(z[:, :, 1::2, :], 3, None, None, device=device)
        elif type == '2c':
            z = self.resamp(x, 3, None, None, device=device)
            if phase == 0:
                y = self.resamp(z[:, :, :, ::2], 0, None, None, device=device)
            else:
                y = self.resamp(hstack((z[:, :, 1:, 1::2].conj().transpose(0, 1, 3, 2),
                                   z[:, :, 0:1, 1::2].conj().transpose(0, 1, 3, 2))).conj().transpose(0, 1, 3, 2), 
                           0, None, None, device=device)
        else:
            print("Invalid argument type")
        return y
    
    
    def extend2(self, x, ru, rd, cl, cr, extmod, device = torch.device("cuda:0")):
        _, _, rx, cx = x.shape
    
        if extmod == 'per':
            I = self.getPerIndices(rx, ru, rd, device=device)
            x = x.to(device)
            y = x[:, :, I, :]
    
            I = self.getPerIndices(cx, cl, cr, device=device)
            y = y[:, :, :, I]
    
            return y
    
        elif extmod == 'qper_row':
            rx2 = round(rx / 2.0)
            y = torch.cat([torch.cat([x[:, :, rx2:rx, cx - cl:cx], x[:, :, 0:rx2, cx - cl:cx]], dim=2), 
                           x, 
                           torch.cat([x[:, :, rx2:rx, 0:cr], x[:, :, 0:rx2, 0:cr]], dim=2)], dim=3)
            I = self.getPerIndices(rx, ru, rd, device=device)
            y = y[:, :, I, :]
            return y
    
        elif extmod == 'qper_col':
            cx2 = int(round(cx / 2.0))
            y = torch.cat([
                torch.cat([x[:, :, rx - ru:rx, cx2:cx], x[:, :, rx - ru:rx, 0:cx2]], dim=3),
                x,
                torch.cat([x[:, :, 0:rd, cx2:cx], x[:, :, 0:rd, 0:cx2]], dim=3)
            ], dim=2)
    
            I = self.getPerIndices(cx, cl, cr, device=device)
            y = y[:, :, :, I]
            return y
    
        else:
            raise ValueError("Invalid input for EXTMOD")

    def getPerIndices(self, lx, lb, le,device = torch.device("cuda:0")):

        # Create ranges directly on GPU
        I1 = torch.arange(lx - lb, lx, device=device)
        I2 = torch.arange(0, lx, device=device)
        I3 = torch.arange(0, le, device=device)
        
    
        # Concatenate tensors on GPU
        I = torch.cat((I1, I2, I3))
    
    
        # Wrap around the indices if they are out of bounds
        if (lx < lb) or (lx < le):
            I = torch.remainder(I, lx)
            I[I == 0] = lx
    
    
        # Clamp operation on GPU
        I = torch.clamp(I, 0, lx - 1)
    
        return I.to(torch.int)
