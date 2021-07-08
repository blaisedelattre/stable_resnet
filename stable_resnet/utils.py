import os
import sys
import re
import shutil
import json
import logging
import glob
import copy
import multiprocessing
from os.path import join
from os.path import exists

from yaml import load, dump
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

import numpy as np
import torch
from torch.distributions import normal, laplace, uniform, bernoulli
from torch.optim.lr_scheduler import _LRScheduler
from advertorch import attacks


def set_default_param_values_and_env_vars(params):
  """Sets up the default param values and environment variables ."""
  # Sets GPU thread settings
  if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
    raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)

  if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
    raise ValueError(
        'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
        params.per_gpu_thread_count)
  # Default to two threads. One for the device compute and the other for
  # memory copies.
  per_gpu_thread_count = params.per_gpu_thread_count or 2
  total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

  cpu_count = multiprocessing.cpu_count()
  if not params.num_inter_threads and params.gpu_thread_mode in [
      'gpu_private', 'gpu_shared'
  ]:
    main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
    params.num_inter_threads = main_thread_count

  # From the total cpu thread count, subtract the total_gpu_thread_count,
  # and then 2 threads per GPU device for event monitoring and sending /
  # receiving tensors
  num_monitoring_threads = 2 * params.num_gpus
  num_private_threads = max(
      cpu_count - total_gpu_thread_count - num_monitoring_threads, 1)
  if params.datasets_num_private_threads == 0:
    params.datasets_num_private_threads = num_private_threads
  return params



def get_global_step_from_ckpt(filename):
  regex = "(?<=ckpt-)[0-9]+"
  return int(re.findall(regex, filename)[-1])


def get_list_checkpoints(train_dir):
  files = glob.glob(join(train_dir, 'model.ckpt-*.pth'))
  files = sorted(files, key=get_global_step_from_ckpt)
  return [filename for filename in files]


def get_checkpoint(train_dir, last_global_step):
  files = get_list_checkpoints(train_dir)
  if not files:
    return None, None
  for filename in files:
    global_step = get_global_step_from_ckpt(filename)
    if last_global_step < global_step:
      return filename, global_step
  return None, None


def get_best_checkpoint(logs_dir):
  best_acc_file = join(logs_dir, "best_accuracy.txt")
  if not exists(best_acc_file):
    raise ValueError("Could not find best_accuracy.txt in {}".format(
            logs_dir))
  with open(best_acc_file) as f:
    content = f.readline().split('\t')
    best_ckpt = content[0]
  best_ckpt_path = glob.glob(
    join(logs_dir[:-5], 'model.ckpt-{}.pth'.format(best_ckpt)))
  return best_ckpt_path[-1], int(best_ckpt)


def remove_training_directory(train_dir):
  """Removes the training directory."""
  try:
    if 'debug' in train_dir:
      logging.info(("Train dir already exist, start_new_model "
                    "set to True and debug mode activated. Train folder "
                    "deleted."))
      shutil.rmtree(train_dir)
    else:
      # to be safe we ask the use to delete the folder manually
      raise RuntimeError(("Train dir already exist and start_new_model "
                          "set to True. To restart model from scratch, "
                          "delete the directory."))
  except:
    logging.error("Failed to delete directory {} when starting a new "
      "model. Please delete it manually and try again.".format(train_dir))
    sys.exit()


class MessageBuilder:

  def __init__(self):
    self.msg = []

  def add(self, name, values, align=">", width=0, format=None):
    if name:
      metric_str = "{}: ".format(name)
    else:
      metric_str = ""
    values_str = []
    if type(values) != list:
      values = [values]
    for value in values:
      if format:
        values_str.append("{value:{align}{width}{format}}".format(
          value=value, align=align, width=width, format=format))
      else:
        values_str.append("{value:{align}{width}}".format(
          value=value, align=align, width=width))
    metric_str += '/'.join(values_str)
    self.msg.append(metric_str)

  def get_message(self):
    message = " | ".join(self.msg)
    self.clear()
    return message

  def clear(self):
    self.msg = []


def setup_logging(verbosity):
  level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
    'INFO': 20, 'WARN': 30
  }[verbosity]
  format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
  logging.basicConfig(level=level, format=format_, datefmt='%H:%M:%S')


class Params:

  def __init__(self):
    self._params = {}

  def add(self, name, value):
    """Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    """
    # Keys in kwargs are unique, but 'name' could the name of a pre-existing
    # attribute of this object.  In that case we refuse to use it as a
    # hyperparameter name.
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    setattr(self, name, value)
    self._params[name] = value

  def override(self, key, new_values):
    """override a parameter in a Params instance."""
    if not isinstance(new_values, dict):
      setattr(self, key, new_values)
      self._params[key] = new_values
      return
    obj = getattr(self, key)
    for k, v in new_values.items():
      obj[k] = v
    setattr(self, key, obj)
    self._params[key] = obj

  def values(self):
    return self._params

  def to_json(self, indent=None, separators=None, sort_keys=False):
    """Serializes the hyperparameters into JSON.
    Args:
      indent: If a non-negative integer, JSON array elements and object members
        will be pretty-printed with that indent level. An indent level of 0, or
        negative, will only insert newlines. `None` (the default) selects the
        most compact representation.
      separators: Optional `(item_separator, key_separator)` tuple. Default is
        `(', ', ': ')`.
      sort_keys: If `True`, the output dictionaries will be sorted by key.
    Returns:
      A JSON string.
    """
    return json.dumps(
        self.values(),
        indent=indent,
        separators=separators,
        sort_keys=sort_keys)


def load_params(config_file, config_name, override_params=None):
  params = Params()
  with open(config_file) as f:
    data = load(f, Loader=Loader)
  for k, v in data[config_name].items():
    params.add(k, v)
  if override_params:
    # logging.info('Overriding parameters of configuration file')
    params_to_override = json.loads(override_params)
    for key, value in params_to_override.items():
      params.override(key, value)
  return params


def get_scheduler(optimizer, lr_scheduler, lr_scheduler_params):
  """Return a learning rate scheduler
  schedulers. See https://pytorch.org/docs/stable/optim.html for more details.
  """
  if lr_scheduler == 'piecewise_constant':
    raise NotImplementedError
  elif lr_scheduler == 'step_lr':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, **lr_scheduler_params)
  elif lr_scheduler == 'multi_step_lr':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer, **lr_scheduler_params)
  elif lr_scheduler == 'exponential_lr':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
      optimizer, **lr_scheduler_params)
  elif lr_scheduler == 'reduce_lr_on_plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, **lr_scheduler_params)
  elif lr_scheduler == 'cyclic_lr':
    scheduler = torch.optim.lr_scheduler.CyclicLR(
      optimizer, **lr_scheduler_params)
  elif lr_scheduler == 'lambda_lr':
    gamma = lr_scheduler_params['gamma']
    decay_every_epoch = lr_scheduler_params['decay_every_epoch']
    warmup = lr_scheduler_params['warmup']
    def lambda_lr(epoch):
      if epoch < warmup:
        return epoch / warmup
      return gamma ** (int(epoch // decay_every_epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer, lr_lambda=lambda_lr)
  else:
    raise ValueError("scheduler was not recognized")
  return scheduler


def get_optimizer(optimizer, opt_args, init_lr, weight_decay, params):
  """Returns the optimizer that should be used based on params."""
  if optimizer == 'sgd':
    opt = torch.optim.SGD(
      params, lr=init_lr, weight_decay=weight_decay, **opt_args)
  elif optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(
      params, lr=init_lr, weight_decay=weight_decay, **opt_args)
  elif optimizer == 'adam':
    opt = torch.optim.Adam(
      params, lr=init_lr, weight_decay=weight_decay, **opt_args)
  elif optimizer == 'rmsproptf':
    # we compute the l2 loss manualy without bn params
    opt = RMSpropTF(params, lr=init_lr, weight_decay=0, **opt_args)
  else:
    raise ValueError("Optimizer was not recognized")
  return opt


def get_attack(model, num_classes, attack_name, attack_params):
  if attack_name == 'carlini':
    attack = attacks.CarliniWagnerL2Attack(model, num_classes, **attack_params)
  elif attack_name == 'elasticnet':
    attack = attacks.ElasticNetL1Attack(model, num_classes, **attack_params)
  elif attack_name == 'pgd':
    norm = attack_params['norm']
    del attack_params['norm']



def get_attack(model, num_classes, attack_name, attack_params):
  if attack_name == 'carlini':
    attack = attacks.CarliniWagnerL2Attack(model, num_classes, **attack_params)
  elif attack_name == 'elasticnet':
    attack = attacks.ElasticNetL1Attack(model, num_classes, **attack_params)
  elif attack_name == 'pgd':
    norm = attack_params['norm']
    del attack_params['norm']
    if norm == 'inf':
      attack = attacks.LinfPGDAttack(model, **attack_params)
    elif norm == 'l1':
      attack = attacks.SparseL1PGDAttack(model, **attack_params)
    elif norm == 'l2':
      attack = attacks.L2PGDAttack(model, **attack_params)
    else:
      raise ValueError("Norm not recognized for PGD attack.")
  elif attack_name == 'fgsm':
    attack = GradientSignAttack(model, **attack_params)
  else:
    raise ValueError("Attack name not recognized for adv training.")
  return attack



class Noise:

  def __init__(self, params):
    self.params = params
    self.noise_distribution = self.params.noise_distribution
    if self.noise_distribution not in ('normal', 'uniform'):
      raise ValueError(
        "Noise distribution should be 'normal' or 'uniform'.")

  def _sample_uniform(self, x):
    dim = np.product(x.shape[1:])
    radius = torch.rand((len(x), 1), device=x.device) ** (1 / dim)
    radius *= np.sqrt(dim + 2)
    noise = torch.randn(x.shape, device=x.device).reshape(len(x), -1)
    noise = noise / torch.norm(noise, dim=1, p=2, keepdim=True) * radius
    return noise.reshape(x.shape)

  def __call__(self, x):
    if self.noise_distribution == 'normal':
      noise = torch.randn(x.shape, device=x.device)
    elif self.noise_distribution == 'uniform':
      noise  = self._sample_uniform(x).to(x.device)
    return noise



class EMA:

  def __init__(self, mu):
    self.mu = mu
    self.shadow = {}

  def state_dict(self):
    return copy.deepcopy(self.shadow)

  def __len__(self):
    return len(self.shadow)

  def __call__(self, module, step=None):
    if step is None:
      mu = self.mu
    else:
      # see: tensorflow doc ExponentialMovingAverage
      mu = min(self.mu, (1. + step) / (10 + step))
    for name, x in module.state_dict().items():
      if name in self.shadow:
        new_average = (1.0 - mu) * x + mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
      else:
        self.shadow[name] = x.clone()


class GradualWarmupScheduler(_LRScheduler):
  """ Gradually warm-up(increasing) learning rate in optimizer.
  Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
  Args:
    optimizer (Optimizer): Wrapped optimizer.
    total_epoch: target learning rate is reached at total_epoch, gradually
    after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
  """
  def __init__(self, optimizer, total_epoch, after_scheduler):
    self.total_epoch = total_epoch
    self.after_scheduler = after_scheduler
    self.finished = False
    super(GradualWarmupScheduler, self).__init__(optimizer)

  def get_lr(self):
    if self.last_epoch >= self.total_epoch:
      if not self.finished:
        self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
        self.finished = True
      # return self.after_scheduler.get_last_lr()
      return self.after_scheduler.get_lr()
    return [base_lr * (float(self.last_epoch + 1) / self.total_epoch) for base_lr in self.base_lrs]

  def step(self, epoch=None, metrics=None):
    if self.finished and self.after_scheduler:
      if epoch is None:
        self.after_scheduler.step(None)
      else:
        self.after_scheduler.step(epoch - self.total_epoch)
      # self._last_lr = self.after_scheduler.get_last_lr()
      self._last_lr = self.after_scheduler.get_lr()
    else:
      return super(GradualWarmupScheduler, self).step(epoch)









