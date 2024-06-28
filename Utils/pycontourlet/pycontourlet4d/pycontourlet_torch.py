from numpy import *
from scipy import signal
from .dfilters import dfilters
from .modulate2 import modulate2
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
#Issue if C compiler not found
try:
    from .resamp4c import resamp4c
except:
    from .resamp4 import resamp4c
import torch
from torch.nn import functional as F
from scipy.ndimage import zoom

import pdb

def batch_multi_channel_pdfbdec(x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3, 3], device=torch.device("cpu")):
    """Multi-channel pyramidal directional filter bank decomposition
     for a batch of images.

        Parameters
        ----------
        x : 4D Tensor
            Tensor in the following dimension:
            (batch_size, channel_size, image_width, image_height)
        pfilt: str, default="maxflat"
            Filter name for the pyramidal decomposition step
        dfilt: str, default="dmaxflat7"
            Filter name for the directional decomposition step
        n_levs : list of int, default=[0, 3, 3, 3]
             The numbers of DFB (Directional Filter Bank) decomposition levels at
             each pyramidal level from coarse to fine-scale.

             In each pyramidal level, there is one number of DFB decomposition
             levels is denoted as `l`, resulting in `2^l` wedge-shaped subbands
             in the frequency domain.  0 denoted a 2-D wavelet decomposition,
             resulting a coarse approximation and three bandpass directional
             subbands in scale 4.

             For example:
                 mode = 'resize'
                 n_channels = 3
                 n_levs = [0, 3, 3, 3]
                 num_subbands = [(1+3)*3, (2^3)*3, (2^3)*3, (2^3)*3]
                              = [12, 24, 24, 24]
        device : torch.device, default=torch.device("cpu")
            The device to store and compute Torch tensor.
            Either "cpu" or "cuda".

        Returns
        -------
        coefs : dictionary of 4D numpy array
            Dictionary containing the coefficients obtained from the
            decomposition, with each level as the key. The index key
            starts is the numpy array shape.
            The dictionary is in the following format:
                ```coefs[level]```
            Here's an example with a batch of 2 images with 3 channels,
            and the following settings:
                >>> x.shape
                torch.Size([2, 3, 224, 224])
                >>> y = batch_multi_channel_pdfbdec(x, n_levs=[0, 3, 3, 3])
            This will yield:
                >>> y[(2, 3, 14, 14)].shape
                (2, 3, 14, 14)
                >>> y[(2, 3, 14, 28)].shape
                (2, 3, 14, 28)
                >>> y[(2, 3, 28, 14)].shape
                (2, 3, 28, 14)
                >>> y[(2, 3, 56, 28)].shape
                (2, 3, 56, 28)
                >>> y[(2, 3, 56, 112)].shape
                (2, 3, 56, 112)
                >>> y[(2, 3, 112, 56)].shape
                (2, 3, 112, 56)
    """
    if len(nlevs) == 0:
        y = [x]
    else:
        # Get the pyramidal filters from the filter name
        h, g = pfilters(pfilt, device=device)
        if nlevs[-1] != 0:
            # Laplacian decomposition
            xlo, xhi = lpdec(x.to(device), h, g, device=device)
            # DFB on the bandpass image
            if dfilt in ['pkva6', 'pkva8', 'pkva12', 'pkva']:
                # Use the ladder structure (which is much more efficient)
                xhi_dir = dfbdec_l(xhi, dfilt, nlevs[-1], device=device)
            else:
                # General case
                xhi_dir = dfbdec(xhi, dfilt, nlevs[-1], device=device)
        else:
            # Special case: nlevs(end) == 0
            # Perform one-level 2-D critically sampled wavelet filter bank                       
            xlo, xLH, xHL, xHH = wfb2dec(x.to(device), h, g, device=device)
            xhi_dir = [xLH, xHL, xHH]

        # Recursive call on the low band
        ylo = batch_multi_channel_pdfbdec(xlo, pfilt, dfilt, nlevs[:-1], device=device)

        # Add bandpass directional subbands to the final output
        y = ylo + [xhi_dir]
        
    return y

def dfbdec(x, fname, n, device=torch.device("cpu")):
    """ DFBDEC   Directional Filterbank Decomposition

    y = dfbdec(x, fname, n)

    Input:
    x:      input image
    fname:  filter name to be called by DFILTERS
    n:      number of decomposition tree levels

    Output:
    y:      subband images in a cell vector of length 2^n

    Note:
    This is the general version that works with any FIR filters

    See also: DFBREC, FBDEC, DFILTERS"""
    if (n != round(n)) or (n < 0):
        print('Number of decomposition levels must be a non-negative integer')

    if n == 0:
        # No decomposition, simply copy input to output
        y = [None]
        y[0] = x.clone().to(device)
        return y

    # Get the diamond-shaped filters
    h0, h1 = dfilters(fname, 'd')

    # Fan filters for the first two levels
    # k0: filters the first dimension (row)
    # k1: filters the second dimension (column)
    k0 = modulate2(h0, 'c', None)
    k1 = modulate2(h1, 'c', None)
    # Tree-structured filter banks
    if n == 1:
        # Simplest case, one level
        y = [[None]] * 2
        y[0], y[1] = fbdec(x, k0, k1, 'q', '1r', 'per', device=device)
    else:
        # For the cases that n >= 2
        # First level
        x0, x1 = fbdec(x, k0, k1, 'q', '1r', 'per', device=device)
        # Second level
        y = [[None]] * 4
        y[0], y[1] = fbdec(x0, k0, k1, 'q', '2c', 'qper_col', device=device)
        y[2], y[3] = fbdec(x1, k0, k1, 'q', '2c', 'qper_col', device=device)
        # Fan filters from diamond filters
        f0, f1 = ffilters(h0, h1, device=device)
        # Now expand the rest of the tree
        for l in range(3, n + 1):
            # Allocate space for the new subband outputs
            y_old = y[:]
            y = [[None]] * 2**l
            # The first half channels use R1 and R2
            for k in range(0, 2**(l - 2)):
                i = k % 2
                y[2 * k], y[2 * k + 1] = fbdec(y_old[k],
                                               f0[i], f1[i], 'pq', i, 'per', device=device)
            # The second half channels use R3 and R4
            for k in range(2**(l - 2), 2**(l - 1)):
                i = (k % 2) + 2
                y[2 * k], y[2 * k + 1] = fbdec(y_old[k],
                                               f0[i], f1[i], 'pq', i, 'per', device=device)
    # Back sampling (so that the overal sampling is separable)
    # to enhance visualization
    y = backsamp(y, device=device)
    # Flip the order of the second half channels
    y[2**(n - 1)::] = y[::-1][:2**(n - 1)]

    return y

def pfilters(fname, device=torch.device("cpu")):
    """ PFILTERS Generate filters for the laplacian pyramid

    Input:
    fname : Name of the filter, including the famous '9-7' filters.

    Output:
    h, g: 1D filters (lowpass for analysis and synthesis, respectively)
    for separable pyramid"""

    if fname == "9/7" or fname == "9-7":
        h = torch.tensor([0.037828455506995, -0.023849465019380, -0.11062440441842, 0.37740285561265], device=device)
        h = torch.cat((h, torch.tensor([0.85269867900940], device=device), h.flip(0)))

        g = torch.tensor([-0.064538882628938, -0.040689417609558, 0.41809227322221], device=device)
        g = torch.cat((g, torch.tensor([0.78848561640566], device=device), g.flip(0)))

    elif fname == "maxflat":
        M1 = 1 / torch.sqrt(torch.tensor(2.0, device=device))
        M2 = M1
        k1 = 1 - torch.sqrt(torch.tensor(2.0, device=device))
        k2 = M1
        k3 = k1
        h = torch.tensor([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3], device=device) * M1
        h = torch.cat((h, h[:-1].flip(0)))

        g = torch.tensor([-0.125 * k1 * k2 * k3, 0.25 * k1 * k2, -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                         1 + 0.5 * k1 * k2], device=device) * M2
        g = torch.cat((g, g[:-1].flip(0)))
        # Normalize
        h = h * torch.sqrt(torch.tensor(2.0, device=device))
        g = g * torch.sqrt(torch.tensor(2.0, device=device))

    elif fname == "5/3" or fname == "5-3":
        h = torch.tensor([-1, 2, 6, 2, -1], device=device) / (4 * torch.sqrt(torch.tensor(2.0, device=device)))
        g = torch.tensor([1, 2, 1], device=device) / (2 * torch.sqrt(torch.tensor(2.0, device=device)))

    elif fname == "burt" or fname == "Burt":
        h = torch.tensor([0.6, 0.25, -0.05], device=device)
        h = torch.sqrt(torch.tensor(2.0, device=device)) * torch.cat((h[1:].flip(0), h))

        g = torch.tensor([17.0 / 28, 73.0 / 280, -3.0 / 56, -3.0 / 280], device=device)
        g = torch.sqrt(torch.tensor(2.0, device=device)) * torch.cat((g[1:].flip(0), g))

    elif fname == "pkva":
        # filters from the ladder structure
        # Allpass filter for the ladder structure network
        beta = ldfilter(fname).to(device)

        lf = len(beta)
        n = float(lf) / 2

        if n != torch.floor(torch.tensor(n, device=device)):
            raise ValueError("The input allpass filter must be even length")

        # beta(z^2)
        beta2 = torch.zeros(2 * lf - 1, device=device)
        beta2[::2] = beta

        # H(z)
        h = beta2.clone()
        h[int(2 * n - 1)] = h[int(2 * n - 1)] + 1
        h = h / 2

        # G(z)
        g = -torch.nn.functional.conv1d(beta2.view(1, 1, -1), h.view(1, 1, -1)).view(-1)
        g[int(4 * n - 2)] = g[int(4 * n - 2)] + 1
        g[1:-1:2] = -g[1:-1:2]

        # Normalize
        h = h * torch.sqrt(torch.tensor(2.0, device=device))
        g = g * torch.sqrt(torch.tensor(2.0, device=device))

    else:
        raise ValueError("Filter name not recognized")

    return h, g


def lpdec(x, h, g, device=torch.device("cpu")):
    """ LPDEC   Laplacian Pyramid Decomposition

    [c, d] = lpdec(x, h, g)

    Input:
    x:      input image
    h, g:   two lowpass filters for the Laplacian pyramid

    Output:
    c:      coarse image at half size
    d:      detail image at full size

    See also:   LPREC, PDFBDEC"""

    # Lowpass filter and downsample
    xlo = sefilter2(x, h, h, 'per', None, device=device)
    c = xlo[:, :, ::2, ::2]
    
    # Compute the residual (bandpass) image by upsample, filter, and subtract
    # Even size filter needs to be adjusted to obtain perfect reconstruction
    adjust = (len(g) + 1) % 2
    xlo = torch.zeros_like(x).to(device)
    xlo[:, :, ::2, ::2] = c
    tmp = sefilter2(xlo, g, g, 'per', adjust * torch.tensor([1, 1], device=device), device=device)
    d = x - tmp

    return c, d


def mod(a, b):
    """Custom mod function to replicate NumPy's mod behavior"""
    return a % b


def sefilter2(x, f1, f2, extmod='per', shift=None, device=torch.device("cpu")):
    """SEFILTER2   2D separable filtering with extension handling
    y = sefilter2(x, f1, f2, [extmod], [shift])

    Input:
    x:      input image
    f1, f2: 1-D filters in each dimension that make up a 2D separable filter
    extmod: [optional] extension mode (default is 'per')
    shift:  [optional] specify the window over which the
    convolution occurs. By default shift = [0; 0].

    Output:
    y:      filtered image of the same size as the input image:
    Y(z1,z2) = X(z1,z2)*F1(z1)*F2(z2)*z1^shift(1)*z2^shift(2)

    Note:
    The origin of the filter f is assumed to be floor(size(f)/2) + 1.
    Amount of shift should be no more than floor((size(f)-1)/2).
    The output image has the same size with the input image.

    See also: EXTEND2, EFILTER2"""

    if shift is None:
        shift = torch.tensor([[0], [0]], dtype=torch.float32, device=device)
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
    lf2 = (len(f2) - 1) / 2.0
    y = extend2(x, floor(torch.tensor(lf1)) + shift[0], ceil(torch.tensor(lf1)) - shift[0],
                floor(torch.tensor(lf2)) + shift[1], ceil(torch.tensor(lf2)) - shift[1], extmod, device=device)

    # Separable filter
    filters = f1 @ f2.T
    filters = filters.unsqueeze(0).unsqueeze(0)
    filters = filters.repeat(y.shape[1], 1, 1, 1)
    
    y = torch.nn.functional.conv2d(y, filters, groups=y.size(1))
    
    return y


def floor(tensor):
    return torch.floor(tensor).long()

def ceil(tensor):
    return torch.ceil(tensor).long()

# def mod(a, b):
#     return a % b

def getPerIndices(lx, lb, le, device=torch.device("cpu")):
    """Get periodic indices for extension."""
    lb = lb.item() if isinstance(lb, torch.Tensor) else lb
    le = le.item() if isinstance(le, torch.Tensor) else le
    
    I = torch.cat((torch.arange(lx - lb, lx, device=device), 
                   torch.arange(0, lx, device=device), 
                   torch.arange(0, le, device=device)))

    # Wrap around the indices if they are out of bounds
    I = torch.remainder(I, lx)
    I = torch.where(I < 0, I + lx, I)  # Handle negative indices
    I = torch.clamp(I, 0, lx - 1)  # Ensure indices are within bounds

    return I.to(torch.int)


def extend2(x, ru, rd, cl, cr, extmod, device=torch.device("cpu")):
    """ EXTEND2   2D extension
    y = extend2(x, ru, rd, cl, cr, extmod)

    Input:
    x:  input image
    ru, rd: amount of extension, up and down, for rows
    cl, cr: amount of extension, left and right, for column
    extmod: extension mode.  The valid modes are:
    'per':      periodized extension (both direction)
    'qper_row': quincunx periodized extension in row
    'qper_col': quincunx periodized extension in column

    Output:
    y:  extended image

    See also:   FBDEC"""
    
    _, _, rx, cx = x.shape

    if extmod == 'per':
        I = getPerIndices(rx, ru, rd, device=device)
        y = x[:, :, I, :]

        I = getPerIndices(cx, cl, cr, device=device)
        y = y[:, :, :, I]

        return y

    elif extmod == 'qper_row':
        rx2 = int(round(rx / 2.0))
        y = torch.cat([torch.cat([x[:, :, rx2:rx, cx - cl:cx], x[:, :, 0:rx2, cx - cl:cx]], dim=2), 
                       x, 
                       torch.cat([x[:, :, rx2:rx, 0:cr], x[:, :, 0:rx2, 0:cr]], dim=2)], dim=3)
        I = getPerIndices(rx, ru, rd, device=device)
        y = y[:, :, I, :]
        return y

    elif extmod == 'qper_col':
        cx2 = int(round(cx / 2.0))
        y = torch.cat([
            torch.cat([x[:, :, rx - ru:rx, cx2:cx], x[:, :, rx - ru:rx, 0:cx2]], dim=3),
            x,
            torch.cat([x[:, :, 0:rd, cx2:cx], x[:, :, 0:rd, 0:cx2]], dim=3)
        ], dim=2)

        I = getPerIndices(cx, cl, cr, device=device)
        y = y[:, :, :, I]
        return y

    else:
        raise ValueError("Invalid input for EXTMOD")


def fbdec(x, h0, h1, type1, type2, extmod, device=torch.device("cpu")):
    """ FBDEC   Two-channel 2D Filterbank Decomposition

    [y0, y1] = fbdec(x, h0, h1, type1, type2, [extmod])

    Input:
    x:  input image
    h0, h1: two decomposition 2D filters
    type1:  'q', 'p' or 'pq' for selecting quincunx or parallelogram
    downsampling matrix
    type2:  second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QDOWN and PDOWN
    If type1 == 'pq' then same as 'p' except that
    the paralellogram matrix is replaced by a combination
    of a  resampling and a quincunx matrices
    extmod: [optional] extension mode (default is 'per')

    Output:
    y0, y1: two result subband images

    Note:       This is the general implementation of 2D two-channel
    filterbank

    See also:   FBDEC_SP """

    # For parallelogram filterbank using quincunx downsampling, resampling is
    # applied before filtering
    if type1 == 'pq':
        x = resamp(x, type2, None, extmod, device=device)

    # Stagger sampling if filter is odd-size (in both dimensions)
    if torch.all(mod(torch.tensor(h1.shape), 2)):
        shift = torch.tensor([[-1], [0]], device=device)

        # Account for the resampling matrix in the parallelogram case
        if type1 == 'p':
            R = [None] * 4
            R[0] = torch.tensor([[1, 1], [0, 1]], device=device)
            R[1] = torch.tensor([[1, -1], [0, 1]], device=device)
            R[2] = torch.tensor([[1, 0], [1, 1]], device=device)
            R[3] = torch.tensor([[1, 0], [-1, 1]], device=device)
            shift = torch.mm(R[type2], shift)
    else:
        shift = torch.tensor([[0], [0]], device=device)

    # Extend, filter and keep the original size
    y0 = efilter2(x, h0, extmod, None, device)
    y1 = efilter2(x, h1, extmod, shift, device)

    # Downsampling
    if type1 == 'q':
        # Quincunx downsampling
        y0 = qdown(y0, type2, extmod, None, device)
        y1 = qdown(y1, type2, extmod, None, device)
    elif type1 == 'p':
        # Parallelogram downsampling
        y0 = pdown(y0, type2, extmod, None, device)
        y1 = pdown(y1, type2, extmod, None, device)
    elif type1 == 'pq':
        # Quincunx downsampling using the equivalent type
        pqtype = ['1r', '2r', '2c', '1c']
        y0 = qdown(y0, pqtype[type2], extmod, None, device)
        y1 = qdown(y1, pqtype[type2], extmod, None, device)
    else:
        raise ValueError("Invalid input type1")

    return y0, y1

def efilter2(x, f, extmod='per', shift=None, device=torch.device("cpu")):
    """EFILTER2   2D Filtering with edge handling (via extension)

    y = efilter2(x, f, [extmod], [shift])

    Input:
    x:  input image
    f:  2D filter
    extmod: [optional] extension mode (default is 'per')
    shift:  [optional] specify the window over which the
    convolution occurs. By default shift = [0; 0].

    Output:
    y:  filtered image that has:
    Y(z1,z2) = X(z1,z2)*F(z1,z2)*z1^shift(1)*z2^shift(2)

    Note:
    The origin of filter f is assumed to be floor(size(f)/2) + 1.
    Amount of shift should be no more than floor((size(f)-1)/2).
    The output image has the same size with the input image.

    See also:   EXTEND2, SEFILTER2"""

    if shift is None:
        shift = torch.tensor([[0], [0]], dtype=torch.float32, device=device)

    # Ensure filters are on the correct device
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.float32, device=device)
    else:
        f = f.clone().detach().to(device)

    # Make sure filters are in the correct format
    if f.ndim < 2:
        f = f.view(-1, 1)

    # Periodized extension
    if f.ndim == 2:
        sf = (f.size(0) - 1) / 2.0, (f.size(1) - 1) / 2.0
    else:
        sf = (f.size(0) - 1) / 2.0, 0.0

    ru = int(floor(torch.tensor(sf[0])) + shift[0][0])
    rd = int(ceil(torch.tensor(sf[0])) - shift[0][0])
    cl = int(floor(torch.tensor(sf[1])) + shift[1][0])
    cr = int(ceil(torch.tensor(sf[1])) - shift[1][0])
    xext = extend2(x, ru, rd, cl, cr, extmod, device)

    # Convolution and keep the central part that has the size as the input
    if f.ndim < 2:
        inputs = xext.to(device).float()  # Ensure input is float
        filters = f[:, None].float().to(device)  # Ensure filter is float
        filters = filters[None, None, :, :]
        filters = torch.repeat_interleave(filters, x.size(1), dim=0)
        y = torch.nn.functional.conv2d(inputs, filters, groups=inputs.size(1)).permute(0, 1, 3, 2)
    else:
        inputs = xext.to(device).float()  # Ensure input is float
        filters = f.float().to(device)  # Ensure filter is float
        filters = filters[None, None, :, :]
        filters = torch.repeat_interleave(filters, x.size(1), dim=0)
        y = torch.nn.functional.conv2d(inputs, filters, groups=inputs.size(1))

    return y




def qdown(x, type, extmod, phase, device=torch.device("cpu")):
    """QDOWN   Quincunx Downsampling

    y = qdown(x, [type], [extmod], [phase])

    Input:
    x:  input image
    type:   [optional] one of {'1r', '1c', '2r', '2c'} (default is '1r')
        '1' or '2' for selecting the quincunx matrices:
            Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
        'r' or 'c' for suppressing row or column
    phase:  [optional] 0 or 1 for keeping the zero- or one-polyphase
        component, (default is 0)

    Output:
    y:  qunincunx downsampled image

    See also: QPDEC"""

    if type is None:
        type = '1r'

    if phase is None:
        phase = 0

    if type == '1r':
        z = resamp(x, 1, None, extmod, device=device)
        if phase == 0:
            y = resamp(z[:, :, ::2, :], 2, None, extmod, device=device)
        else:
            y = resamp(hstack([z[:, :, 1::2, 1:], z[:, :, 1::2, 0:1]]), 2, None, extmod, device=device)

    elif type == '1c':
        z = resamp(x, 2, None, extmod, device=device)
        if phase == 0:
            y = resamp(z[:, :, :, ::2], 1, None, extmod, device=device)
        else:
            y = resamp(z[:, :, :, 1::2], 1, None, extmod, device=device)
            
    elif type == '2r':
        z = resamp(x, 0, None, extmod, device=device)
        if phase == 0:
            y = resamp(z[:, :, ::2, :], 3, None, extmod, device=device)
        else:
            y = resamp(z[:, :, 1::2, :], 3, None, extmod, device=device)
            
    elif type == '2c':
        z = resamp(x, 3, None, extmod, device=device)
        if phase == 0:
            y = resamp(z[:, :, :, ::2], 0, None, extmod, device=device)
        else:
            y = resamp(hstack([z[:, :, 1:, 1::2].transpose(0, 1, 3, 2),
                               z[:, :, 0:1, 1::2].transpose(0, 1, 3, 2)]).transpose(0, 1, 3, 2), 
                       0, None, extmod, device=device)
    else:
        raise ValueError("Invalid argument type")
        
    return y


def resamp(x, type_, shift, extmod, device=torch.device("cpu")):
    """ RESAMP   Resampling in 2D filterbank

    y = resamp(x, type, [shift, extmod])

    Input:
    x:  input image
    type: one of {0,1,2,3} (see note)

    shift:  [optional] amount of shift (default is 1)
    extmod: [optional] extension mode (default is 'per').
    Other options are:

    Output:
    y:  resampled image.

    Note:
    The resampling matrices are:
            R1 = [1, 1;  0, 1];
            R2 = [1, -1; 0, 1];
            R3 = [1, 0;  1, 1];
            R4 = [1, 0; -1, 1];

    For type 1 and type 2, the input image is extended (for example
    periodically) along the vertical direction;
    while for type 3 and type 4 the image is extended along the
    horizontal direction.

    Calling resamp(x, type, n) which n is positive integer is equivalent
    to repeatly calling resamp(x, type) n times.

    Input shift can be negative so that resamp(x, 1, -1) is the same
    with resamp(x, 2, 1)"""

    # Convert to np.float32
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
        raise ValueError("The second input (type_) must be one of {0, 1, 2, 3}")

    return y

def ffilters(h0, h1, device=torch.device("cpu")):
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

def backsamp(y, device=torch.device("cpu")):
    """ BACKSAMP
    Backsampling the subband images of the directional filter bank

       y = backsamp(y)

     Input and output are cell vector of dyadic length

     This function is called at the end of the DFBDEC to obtain subband images
     with overall sampling as diagonal matrices

     See also: DFBDEC"""

    # Number of decomposition tree levels
    n = int(log2(len(y)))

    if (n != round(n)) or (n < 1):
        print("Input must be a cell vector of dyadic length")
    if n == 1:
        # One level, the decomposition filterbank shoud be Q1r
        # Undo the last resampling (Q1r = R2 * D1 * R3)
        for k in range(0, 2):
            y[k] = resamp(y[k], 3, None, None, device=device)
            y[k][:, 0::2] = resamp(y[k][:, 0::2], 0, None, None, device=device)
            y[k][:, 1::2] = resamp(y[k][:, 1::2], 0, None, None, device=device)

    elif n > 2:
        N = 2**(n - 1)
        for k in range(0, 2**(n - 2)):
            shift = 2 * (k + 1) - (2**(n - 2) + 1)
            # The first half channels
            y[2 * k] = resamp(y[2 * k], 2, shift, None, device=device)
            y[2 * k + 1] = resamp(y[2 * k + 1], 2, shift, None, device=device)
            # The second half channels
            y[2 * k + N] = resamp(y[2 * k + N], 0, shift, None, device=device)
            y[2 * k + 1 + N] = resamp(y[2 * k + 1 + N], 0, shift, None, device=device)

    return y

def wfb2dec(x, h, g, device=torch.device("cpu")):
    """WFB2DEC   2-D Wavelet Filter Bank Decomposition
    
    y = wfb2dec(x, h, g)

    Input:
    x:      input image
    h, g:   lowpass analysis and synthesis wavelet filters

    Output:
    x_LL, x_LH, x_HL, x_HH:   Four 2-D wavelet subbands"""
    
    # Ensure h and g are on the correct device and convert to tensors if necessary
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h, dtype=torch.float32, device=device)
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g, dtype=torch.float32, device=device)

    # Make sure filter in a row vector
    h = h.view(-1, 1)
    g = g.view(-1, 1)

    h0 = h
    len_h0 = len(h0)
    ext_h0 = floor(torch.tensor(len_h0 / 2.0))
    # Highpass analysis filter: H1(z) = -z^(-1) G0(-z)
    len_h1 = len(g)
    c = floor(torch.tensor((len_h1 + 1.0) / 2.0))
    # Shift the center of the filter by 1 if its length is even.
    if mod(len_h1, 2) == 0:
        c = c + 1

    # Highpass filter
    h1 = -g.squeeze() * (-1) ** (torch.arange(1, len_h1 + 1, device=device) - c)
    ext_h1 = len_h1 - c + 1
    pdb.set_trace()
    # Debugging print statements
    print(f"Initial input size: {x.size()}")
    print(f"Lowpass filter h0 size: {h0.size()}")
    print(f"Highpass filter h1 size: {h1.size()}")
    
    # Row-wise filtering
    x_L = rowfiltering(x, h0, ext_h0, device=device)
    x_L = x_L[:, :, :, ::2]  # Downsample columns

    x_H = rowfiltering(x, h1, ext_h1, device=device)
    x_H = x_H[:, :, :, ::2]  # Downsample columns

    # Column-wise filtering
    x_LL = rowfiltering(x_L.permute(0, 1, 3, 2), h0, ext_h0, device=device)
    x_LL = x_LL.permute(0, 1, 3, 2)
    x_LL = x_LL[:, :, ::2, :]  # Downsample rows

    x_LH = rowfiltering(x_L.permute(0, 1, 3, 2), h1, ext_h1, device=device)
    x_LH = x_LH.permute(0, 1, 3, 2)
    x_LH = x_LH[:, :, ::2, :]  # Downsample rows

    x_HL = rowfiltering(x_H.permute(0, 1, 3, 2), h0, ext_h0, device=device)
    x_HL = x_HL.permute(0, 1, 3, 2)
    x_HL = x_HL[:, :, ::2, :]  # Downsample rows

    x_HH = rowfiltering(x_H.permute(0, 1, 3, 2), h1, ext_h1, device=device)
    x_HH = x_HH.permute(0, 1, 3, 2)
    x_HH = x_HH[:, :, ::2, :]  # Downsample rows

    return x_LL, x_LH, x_HL, x_HH



def rowfiltering(x, f, ext1, device=torch.device("cpu")):
    pdb.set_trace()
    """Internal function: Row-wise filtering with border handling"""
    
    ext1 = int(ext1)
    ext2 = int(len(f) - ext1 - 1)
    x = torch.cat([x[:, :, :, -ext1:], x, x[:, :, :, :ext2]], dim=3)
    
    # Prepare inputs and filters for convolution
    inputs = x.conj().permute(0, 1, 3, 2).to(device)
    filters = torch.tensor(f, dtype=torch.float32, device=device).view(1, 1, -1, 1)
    filters = filters.repeat(inputs.shape[1], 1, 1, 1)
    
    # Perform convolution
    y = F.conv2d(inputs, filters, groups=inputs.size(1)).conj().permute(0, 1, 3, 2)

    return y