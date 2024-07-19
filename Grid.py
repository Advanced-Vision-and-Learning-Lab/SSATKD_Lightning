#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:29:33 2024

@author: jarin.ritu
"""

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd

# # Example data
# data = np.random.rand(8, 8)
# labels = [f"label_{i}" for i in range(16)]

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Plot the heatmap
# plt.figure(figsize=(12, 12))
# sns.heatmap(df, annot=False, cmap='Blues')
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Example data: A 1xHW tensor with H=4 and W=6
H, W = 4, 6
tensor = np.random.rand(H, W)

# Plot the tensor
plt.figure(figsize=(6, 4))
plt.imshow(tensor, cmap='gray')
plt.colorbar(label='Intensity')
plt.title('1xHW Tensor Visualization')
plt.show()
