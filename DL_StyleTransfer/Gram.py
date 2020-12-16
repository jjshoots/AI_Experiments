import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict

class Gram(nn.Module):
    def matrix(self, input):
        b, c, h, w = input.shape

        F1 = input.view(b, c, h*w)
        F2 = F1.transpose(1, 2)

        G = torch.bmm(F1, F2)

        G = G.div(h*w)
        return G

    def MSELoss(self, input, target):
        return nn.MSELoss()(self.matrix(input), target)


