
import logging
import random
import math
from os.path import join
from os.path import exists

from .utils import NormalizeLayer

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from PIL import Image


class BaseReader:

  def __init__(self, params, batch_size, num_gpus, is_training):
    self.params = params
    self.num_gpus = num_gpus
    self.num_splits = num_gpus
    self.batch_size = batch_size
    self.is_training = is_training
    self.normalization_mean = None
    self.normalization_std = None
    self.path = join(self.get_data_dir(), self.params.dataset)

  def get_data_dir(self):
    paths = self.params.data_dir.split(':')
    data_dir = None
    for path in paths:
      if exists(join(path, self.params.dataset)):
        data_dir = path
        break
    if data_dir is None:
      raise ValueError("Data directory not found.")
    return data_dir

  def transform(self):
    """Create the transformer pipeline."""
    raise NotImplementedError('Must be implemented in derived classes')

  def load_dataset(self):
    """Load or download dataset."""
    if getattr(self.params, 'job_name', None):
      sampler = torch.utils.data.distributed.DistributedSampler(
        self.dataset, num_replicas=None, rank=None)
    else:
      sampler = None
    loader = DataLoader(self.dataset,
                        batch_size=self.batch_size,
                        num_workers=2*self.num_gpus,
                        shuffle=self.is_training and not sampler,
                        pin_memory=bool(self.num_gpus),
                        sampler=sampler)
    return loader, sampler

  def get_normalize_layer(self):
    assert self.normalization_mean is not None
    assert self.normalization_std is not None
    return NormalizeLayer(
      self.normalization_mean,
      self.normalization_std)



class MNISTReader(BaseReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(MNISTReader, self).__init__(
      params, batch_size, num_gpus, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 28, 28
    self.n_train_files = 60000
    self.n_test_files = 10000
    self.n_classes = 10
    self.img_size = (None, 1, 32, 32)

    transform = self.transform()

    self.dataset = MNIST(path, train=self.is_training,
                         download=False, transform=transform)

  def transform(self):
    transform = Compose([
        transforms.ToTensor()])
    return transform


class CIFARReader(BaseReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(CIFARReader, self).__init__(
      params, batch_size, num_gpus, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.img_size = (None, 3, 32, 32)
    self.use_data_augmentation = self.params.data_augmentation

    self.normalization_mean = (0.4914, 0.4822, 0.4465) 
    self.normalization_std = (0.2023, 0.1994, 0.2010)

  def transform(self):
    if self.is_training:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
      ])
    else:
      transform = transforms.Compose([
        transforms.ToTensor()
      ])
    return transform


class CIFAR10Reader(CIFARReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(CIFAR10Reader, self).__init__(
      params, batch_size, num_gpus, is_training)
    self.n_classes = 10
    transform = self.transform()
    self.dataset = CIFAR10(self.path, train=self.is_training,
                           download=False, transform=transform)


class CIFAR100Reader(CIFARReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(CIFAR100Reader, self).__init__(
      params, batch_size, num_gpus, is_training)
    self.n_classes = 100
    transform = self.transform()
    self.dataset = CIFAR100(self.path, train=self.is_training,
                           download=False, transform=transform)


class IMAGENETReader(BaseReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(IMAGENETReader, self).__init__(
      params, batch_size, num_gpus, is_training)

    # Provide square images of this size. 
    self.image_size = self.params.imagenet_image_size
    if 'efficientnet' in self.params.model:
      self.image_size = {
        'efficientnet-b0': 224,
        'efficientnet-b1': 240,
        'efficientnet-b2': 260,
        'efficientnet-b3': 300,
        'efficientnet-b4': 380,
        'efficientnet-b5': 456,
        'efficientnet-b6': 528,
        'efficientnet-b7': 600,
      }[self.params.model]

    self.eigval = [0.2175, 0.0188, 0.0045]
    self.eigvec = [
          [-0.5675,  0.7192,  0.4009],
          [-0.5808, -0.0045, -0.8140],
          [-0.5836, -0.6948,  0.4203],
      ]

    self.normalize_mean = [0.485, 0.456, 0.406]
    self.normalize_std = [0.229, 0.224, 0.225]

    self.imagenet_normalize = transforms.Normalize(
      mean=self.normalize_mean,
      std=self.normalize_std
    )

    self.height, self.width = self.image_size, self.image_size
    self.n_train_files = 1281167
    self.n_test_files = 50000
    self.n_classes = 1000
    self.img_size = (None, 3, self.height, self.height)

    split = 'train' if self.is_training else 'val'

    if 'efficientnet' in self.params.model:
      transform = self.efficientnet_transform()
    else:
      transform = self.transform()

    self.dataset = ImageNet(self.path, split=split,
                            transform=transform)

  def efficientnet_transform(self):
    if self.is_training:
      transform = Compose([
        efficientnet_utils.EfficientNetRandomCrop(self.image_size),
        transforms.Resize(
          (self.image_size, self.image_size), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        efficientnet_utils.Lighting(0.1, self.eigval, self.eigvec),
      ])
    else:
      transform = Compose([
        efficientnet_utils.EfficientNetCenterCrop(self.image_size),
        transforms.Resize(
          (self.image_size, self.image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
      ])
    return transform

  def transform(self):
    if self.is_training:
      transform = Compose([
        transforms.RandomResizedCrop(self.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
      ])
    else:
      transform = Compose([
        transforms.Resize(self.image_size+32),
        transforms.CenterCrop(self.image_size),
        transforms.ToTensor(),
      ])
    return transform


readers_config = {
  'mnist': MNISTReader,
  'cifar10': CIFAR10Reader,
  'cifar100': CIFAR100Reader,
  'imagenet': IMAGENETReader
}
