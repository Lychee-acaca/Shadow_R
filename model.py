import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os

from model_convnext import fusion_net

from Restormer.restormer_arch import Restormer


class final_net(nn.Module):
    def __init__(self, device_cpu, device_gpu):
        super(final_net, self).__init__()
        self.device_cpu = device_cpu
        self.device_gpu = device_gpu
        self.remove_model = fusion_net()
        self.enhancement_model = Restormer()

    def forward(self, input, scale=0.05):
        input.to(self.device_cpu)
        x = self.remove_model(input)
        input.to(self.device_gpu)
        x_ = (self.enhancement_model(x) * scale + x ) / (1 + scale)
        return x_


