# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 14:32:14 2024

@author: jpeeples
"""

import numpy as np
import numpy.ma as ma
import torch
import pdb


def is_string_like(obj):
    """Return True if *obj* looks like a string
    
    Ported from: https://github.com/pieper/matplotlib/blob/master/lib/matplotlib/cbook.py
    with modification to adapt to Python3.
    """
    if isinstance(obj, str):
        return True
    # numpy strings are subclass of str, ma strings are not
    if ma.isMaskedArray(obj):
        if obj.ndim == 0 and obj.dtype.kind in 'SU':
            return True
        else:
            return False
    try:
        obj + ''
    except:
        return False
    return True

# Define the data type for arrays
DTYPE = np.float32

def resampc_periodic_vect(x, m, n, y, type_, shift, batch, channel):
    # pdb.set_trace()
    for b in range(batch):
       x[b] = torch.roll(x[b],shift,dims=-1)
    return x

def resampc_periodic(x, m, n, y, type_, shift, batch, channel):
    for b in range(batch):
        for c in range(channel):
            for j in range(n):
                # Circular shift in each column
                if type_ == 0:
                    k = (shift * j) % m
                else:
                    k = (-shift * j) % m

                # Convert to non-negative mod if needed
                if k < 0:
                    k += m
                for i in range(m):
                    if k >= m:
                        k -= m

                    y[b, c, i, j] = x[b, c, k, j]

                    k += 1
    return y

def resamp4c(x, type_, shift, extmod):
    # pdb.set_trace()
    """RESAMPC. Resampling along the column

    y = resampc(x, type, shift, extmod)

    Input:
        x:      image that is extendable along the column direction
        type:   either 0 or 1 (0 for shuffling down and 1 for up)
        shift:  amount of shifts (typically 1)
        extmod: extension mode:
         - 'per': periodic
         - 'ref1': reflect about the edge pixels
         - 'ref2': reflect, doubling the edge pixels

    Output:
        y: resampled image with:
           R1 = [1, shift; 0, 1] or R2 = [1, -shift; 0, 1]
    """
    if type_ != 0 and type_ != 1:
        raise ValueError("The second input (type_) must be either 0 or 1")
    
    if shift == 0:
        raise ValueError("The third input (shift) cannot be 0")
    
    if not isinstance(extmod, str):
        raise ValueError("EXTMOD arg must be a string")
    
    y = torch.zeros_like(x)
    
    batch = x.shape[0]
    channel = x.shape[1]
    m = x.shape[2]
    n = x.shape[3]
    
    assert x.shape == y.shape
    
    if extmod == 'per':
        y = resampc_periodic_vect(x, m, n, y, type_, shift, batch, channel)
    else:
        raise ValueError(f"Unsupported extension mode: {extmod}")
                
    return y

# Example usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 2, 2, device=device)
    type_ = 0  # Resampling type
    shift = 1  # Amount of shift
    extmod = 'per'  # Extension mode

    y, y_2= resamp4c(x, type_, shift, extmod)
    print("Input array:")
    print(x)
    print("Resampled array:")
    print(y)
    print("Resampled array (vectorized):")
    print(y_2)
    print((y-y_2).sum())