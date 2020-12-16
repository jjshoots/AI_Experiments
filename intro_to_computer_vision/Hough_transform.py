import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

import torch
from torch import cos, sub, t, unsqueeze
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from PIL import Image

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

output = output.apply_(clip)

# Perform polar hough transform
theta_increment = 0.1

d_range = int(math.sqrt(output.shape[0]**2 + output.shape[1]**2))
theta_range = int(90 / theta_increment)
hough_space = np.zeros([theta_range, d_range])

for x, row in enumerate(output):
    for y, pixel in enumerate(row):
        if pixel == 1:
            for subTheta in range(1, theta_range):
                theta = subTheta * theta_increment
                d = x * math.cos(theta) - y * math.sin(theta)
                hough_space[subTheta][int(d)] += 1

index = np.unravel_index(np.argmax(hough_space, axis=None), hough_space.shape)


plt.imshow(1 - output.squeeze(), cmap='Greys')
plt.show()