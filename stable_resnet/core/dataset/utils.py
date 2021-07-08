
from typing import List
import torch

class NormalizeLayer(torch.nn.Module):
  """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.
    """

  def __init__(self, means: List[float], sds: List[float]):
    """
    :param means: the channel means
    :param sds: the channel standard deviations
    """
    super(NormalizeLayer, self).__init__()
    self.means = torch.tensor(means)
    self.sds = torch.tensor(sds)

  def forward(self, input: torch.tensor):
    (batch_size, num_channels, height, width) = input.shape
    means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
    sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
    means = means.to(input.device)
    sds = sds.to(input.device)
    return (input - means)/sds

