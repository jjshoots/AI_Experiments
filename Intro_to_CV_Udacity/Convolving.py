import torch
import torch.nn as nn
import torch.nn.functional as F

torch.no_grad()

input = torch.randn([1, 1, 10, 10])
kernel = torch.ones([1, 1, 3, 3]) * 2

print(input)
print(kernel)

output = F.conv2d(input, kernel, padding=1)

print(output)