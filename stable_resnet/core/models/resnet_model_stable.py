
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..lipschitz import LipschitzBound


class StableBlock(nn.Module):

  def __init__(self, cin, cout, leaky_slope, h, stride=1):
    super(StableBlock, self).__init__()
    self.leaky_relu = nn.LeakyReLU(
      negative_slope=leaky_slope, inplace=False)
    self.kernel = torch.Tensor(cout, cin, 3, 3)
    self.bias = torch.Tensor(cout)
    self.kernel = nn.Parameter(self.kernel)
    self.bias = nn.Parameter(self.bias)
    self.h = h

    self.lip = LipschitzBound(self.kernel.shape, padding=1)

    # initialize weights and biases
    nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5)) # weight init
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)  # bias init

  def forward(self, x):
    res = F.conv2d(x, self.kernel, bias=self.bias, stride=1, padding=1)
    res = self.leaky_relu(res)
    res = F.conv_transpose2d(res, self.kernel, stride=1, padding=1)
    sv_max = self.lip.compute(self.kernel)
    h = 2 / sv_max**2
    logging.info('h = {}'.format(h))
    out = x + h * res
    return out



class ResNet(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(ResNet, self).__init__()

    self.is_training = is_training
    self.num_classes = num_classes
    self.params = params
    self.config = config = self.params.model_params
    self.depth = config['depth']
    self.leaky_slope = config['leaky_slope']
    self.h = config['discretization_step']
    # self.bn_affine = getattr(config, 'bn_affine', True)

    layers = []
    for _ in range(self.depth):
      layers.append(
        StableBlock(3, 3, self.leaky_slope, self.h))
    self.stable_block =  nn.Sequential(*layers)

    self.last = nn.Linear(3*32*32, num_classes)

  def forward(self, x):
    x = self.stable_block(x)
    # logging.info('x shape {}'.format(x.shape))
    return self.last(x.view(-1, 3*32*32))



