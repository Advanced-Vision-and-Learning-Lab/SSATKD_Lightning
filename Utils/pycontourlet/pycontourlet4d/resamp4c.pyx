import numpy as np
import numpy.ma as ma

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
    
    y = np.zeros(x.shape, dtype=np.float32)
    
    batch = x.shape[0]
    channel = x.shape[1]
    m = x.shape[2]
    n = x.shape[3]
    
    if extmod == 'per':
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

# Example usage
if __name__ == '__main__':
    # Example data
    x = np.random.rand(2, 3, 4, 5).astype(DTYPE)  # Create a random 4D array
    type_ = 0  # Resampling type
    shift = 1  # Amount of shift
    extmod = 'per'  # Extension mode

    # Perform resampling
    y = resamp4c(x, type_, shift, extmod)
    
    print("Input array:")
    print(x)
    print("Resampled array:")
    print(y)
