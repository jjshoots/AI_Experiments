import os
import time
from matplotlib.pyplot import axis

import torch
from torch import t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import GRID_SAMPLE_INTERPOLATION_MODES

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
img = prep(Image.open(img_path))

# create offset images
offsetX_img = torch.roll(img, 1, dims=1)
offsetX_img[:, 0, :] = 0

offsetY_img = torch.roll(img, 1, dims=2)
offsetY_img[:, :, 0] = 0

# compute gradient
gradientX = img - offsetX_img
gradientY = img - offsetY_img
gradient = torch.sqrt(gradientX**2 + gradientY**2)

gradient -= torch.min(gradient)
gradient /= torch.max(gradient)

# concatenate images side by side to show
gradient_img = torch.cat([gradient, img], dim=2)

plt.imshow(gradient_img.permute([1, 2, 0]))
plt.show()