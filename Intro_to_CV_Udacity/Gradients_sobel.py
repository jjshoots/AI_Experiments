import os
import time
from matplotlib.pyplot import axis

import torch
from torch import t, unsqueeze
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

# global variables and settings
torch.no_grad()
img_path = os.path.join(os.path.dirname(__file__), 'imgs/img.jpg')

# preprocessor
prep = transforms.Compose([transforms.ToTensor()])

# read image
img = prep(Image.open(img_path)).unsqueeze(0)

# sobel kernel
kernel = torch.tensor(
    [[-1.0, 0, 1.0],
    [-2.0, 0, 2.0],
    [-1.0, 0, 1.0]])
kernel = kernel.unsqueeze(0)
kernel = torch.cat([kernel, kernel, kernel], dim=0)
kernel = kernel.unsqueeze(0)

# convolve kernel with image
output1 = F.conv2d(img, kernel, padding=1)
output2 = F.conv2d(img, torch.transpose(kernel, dim0=2, dim1=3), padding=1)
output = torch.sqrt(output1**2 + output2**2)
output = output.squeeze()
output -= torch.min(output)
output /= torch.max(output)

def clip(input):
    if input < 0.2:
        return 0
    else:
        return 1

output.apply_(clip)

output = 1 - output
plt.imshow(output.squeeze(), cmap='Greys')
plt.show()