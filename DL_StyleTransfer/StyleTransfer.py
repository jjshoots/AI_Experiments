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

from VGGNet import VGGNet
from Gram import Gram

import numpy as np
import matplotlib.pyplot as plt

import urllib.request

# params
dirname = os.path.dirname(__file__)
img_size = 512
net_path = os.path.join(dirname, 'vgg_conv.pth')
style_img_path = os.path.join(dirname, 'imgs/style.jpg')
content_img_path = os.path.join(dirname, 'imgs/content.jpg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using ', device)
max_iteration = 500

# preprocess
prep = transforms.Compose([transforms.Resize([img_size, img_size]),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])

#postpostprocess
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])

postpb = transforms.Compose([transforms.ToPILImage()])

def postp(tensor):
    t = postpa(tensor)
    t = torch.clamp(t, 0, 1)
    img = postpb(t)
    return img

# import images and preprocess them
style_img_torch = prep(Image.open(style_img_path))
content_img_torch = prep(Image.open(content_img_path))
output_img_torch = content_img_torch.unsqueeze(0).to(device)
output_img_torch.requires_grad = True

style_img_torch = style_img_torch.unsqueeze(0).to(device)
content_img_torch = content_img_torch.unsqueeze(0).to(device)

#get network
vgg = VGGNet().to(device)
vgg.load_state_dict(torch.load(net_path))
for param in vgg.parameters():
    param.requires_grad = False
optimizer = optim.LBFGS([output_img_torch])

# define style and content targets
style_layers = ['r11','r21','r31','r41', 'r51']
content_layers = ['r42']

style_outputs = vgg.forward(style_img_torch, style_layers)
content_outputs = vgg.forward(content_img_torch, content_layers)

style_targets = [Gram().matrix(A).detach() for A in style_outputs]
content_targets = [A.detach() for A in content_outputs]

# weight settings
style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
content_weights = [1e0]

for iter in range(0, max_iteration):
    def closure():
        optimizer.zero_grad()

        passthrough = vgg.forward(output_img_torch, style_layers + content_layers)
        style_passthrough = passthrough[0:len(style_layers)]
        content_passthrough = passthrough[len(style_layers)+1:]

        style_loss = [style_weights[a] * Gram().MSELoss(A, style_targets[a]) for a, A in enumerate(style_passthrough)]
        content_loss = [content_weights[a] * nn.MSELoss()(A, content_targets[a]) for a, A in enumerate(content_passthrough)]

        loss = sum(content_loss + style_loss)
        loss.backward()

        print('Iteration: ', iter, '- Loss: ', loss.item())

        if iter % 100 == 3:
            plt.imshow(postp(output_img_torch.squeeze().cpu()))
            plt.show()

        return loss

    optimizer.step(closure)