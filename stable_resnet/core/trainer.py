
import os
import time
import datetime
import pprint
import socket
import logging
import warnings
import glob
from os import mkdir
from os.path import join
from os.path import exists

import utils
from .models import model_config
from .dataset.readers import readers_config

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.distributions import Categorical, kl_divergence


class Trainer:
  """A Trainer to train a PyTorch."""

  def __init__(self, params):
    """Creates a Trainer.
    """
    # utils.set_default_param_values_and_env_vars(params)
    self.params = params

    # Setup logging & log the version.
    utils.setup_logging(params.logging_verbosity)

    self.job_name = self.params.job_name  # "" for local training
    self.is_distributed = bool(self.job_name)
    self.task_index = self.params.task_index
    self.local_rank = self.params.local_rank
    self.start_new_model = self.params.start_new_model
    self.train_dir = self.params.train_dir
    self.num_gpus = self.params.num_gpus
    if self.num_gpus and not self.is_distributed:
      self.batch_size = self.params.batch_size * self.num_gpus
    else:
      self.batch_size = self.params.batch_size

    # print self.params parameters
    if self.start_new_model and self.local_rank == 0:
      pp = pprint.PrettyPrinter(indent=2, compact=True)
      logging.info(pp.pformat(params.values()))

    if self.local_rank == 0:
      logging.info("PyTorch version: {}.".format(torch.__version__))
      logging.info("NCCL Version {}".format(torch.cuda.nccl.version()))
      logging.info("Hostname: {}.".format(socket.gethostname()))

    if self.is_distributed:
      self.num_nodes = len(params.worker_hosts.split(';'))
      self.world_size = self.num_nodes * self.num_gpus
      self.rank = self.task_index * self.num_gpus + self.local_rank
      dist.init_process_group(
        backend='nccl', init_method='env://',
        timeout=datetime.timedelta(seconds=30))
      if self.local_rank == 0:
        logging.info('World Size={} => Total batch size {}'.format(
          self.world_size, self.batch_size*self.world_size))
      self.is_master = bool(self.rank == 0)
    else:
      self.world_size = 1
      self.is_master = True

    # create a mesage builder for logging
    self.message = utils.MessageBuilder()

    # load reader and model
    self.reader = readers_config[self.params.dataset](
      self.params, self.batch_size, self.num_gpus, is_training=True)

    # load model
    self.model = model_config.get_model_config(
        self.params.model, self.params.dataset, self.params,
        self.reader.n_classes, is_training=True)
    # add normalization as first layer of model
    if self.params.add_normalization:
      # In order to certify radii in original coordinates rather than standardized coordinates, we
      # add the noise _before_ standardizing, which is why we have standardization be the first
      # layer of the classifier rather than as a part of preprocessing as is typical.
      normalize_layer = self.reader.get_normalize_layer()
      self.model = torch.nn.Sequential(normalize_layer, self.model)

    # define DistributedDataParallel job
    self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
    torch.cuda.set_device(params.local_rank)
    self.model = self.model.cuda()
    i = params.local_rank
    self.model = DistributedDataParallel(
      self.model, device_ids=[i], output_device=i)
    if self.local_rank == 0:
      logging.info('Model defined with DistributedDataParallel')

    # define set for saved ckpt
    self.saved_ckpts = set([0])

    # define optimizer
    self.optimizer = utils.get_optimizer(
                       self.params.optimizer,
                       self.params.optimizer_params,
                       self.params.init_learning_rate,
                       self.params.weight_decay,
                       self.model.parameters())

    # define the loss
    self.criterion = torch.nn.CrossEntropyLoss().cuda()

    # define learning rate scheduler
    self.scheduler = utils.get_scheduler(
      self.optimizer, self.params.lr_scheduler,
      self.params.lr_scheduler_params)

    # if start_new_model is False, we restart training
    if not self.start_new_model:
      if self.local_rank == 0:
        logging.info('Restarting training...')
      self._load_state()

    # define Lipschitz regularization module
    if self.params.lipschitz_regularization:
      if self.local_rank == 0:
        logging.info(
          "Lipschitz regularization with decay {}, start after epoch {}".format(
            self.params.lipschitz_decay, self.params.lipschitz_start_epoch))
      self.lipschitz = LipschitzRegularization(
        self.model, self.params, self.reader, self.local_rank)

    # exponential moving average
    self.ema = None
    if getattr(self.params, 'ema', False) > 0:
      self.ema = utils.EMA(self.params.ema)

    # if adversarial training, create the attack class
    if self.params.adversarial_training:
      if self.local_rank == 0:
        logging.info('Adversarial Training')
      attack_params = self.params.adversarial_training_params
      if 'eps_iter' in attack_params.keys() and attack_params['eps_iter'] == -1:
        eps = attack_params['eps']
        n_iter = attack_params['nb_iter']
        attack_params['eps_iter'] = eps / n_iter * 2
        if self.local_rank == 0:
          logging.info('Learning rate for attack: {}'.format(attack_params['eps_iter']))
      self.attack = utils.get_attack(
                      self.model,
                      self.reader.n_classes,
                      self.params.adversarial_training_name,
                      attack_params)


  def _load_state(self):
    # load last checkpoint
    checkpoints = glob.glob(join(self.train_dir, "model.ckpt-*.pth"))
    get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
    checkpoints = sorted(
      [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
    path_last_ckpt = join(self.train_dir, checkpoints[-1])
    self.checkpoint = torch.load(path_last_ckpt)
    self.model.load_state_dict(self.checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(self.checkpoint['scheduler'])
    self.saved_ckpts.add(self.checkpoint['epoch'])
    epoch = self.checkpoint['epoch']
    if self.local_rank == 0:
      logging.info('Loading checkpoint {}'.format(checkpoints[-1]))


  def _save_ckpt(self, step, epoch, final=False):
    """Save ckpt in train directory."""
    freq_ckpt_epochs = self.params.save_checkpoint_epochs
    if (epoch % freq_ckpt_epochs == 0 and self.is_master \
        and epoch not in self.saved_ckpts) \
         or (final and self.is_master):
      ckpt_name = "model.ckpt-{}.pth".format(step)
      ckpt_path = join(self.train_dir, ckpt_name)
      if exists(ckpt_path): return 
      self.saved_ckpts.add(epoch)
      state = {
        'epoch': epoch,
        'global_step': step,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict()
      }
      if self.ema is not None:
        state['ema'] = self.ema.state_dict()
      logging.info("Saving checkpoint '{}'.".format(ckpt_name))
      torch.save(state, ckpt_path)


  def run(self):
    """Performs training on the currently defined Tensorflow graph.
    """
    # reset the training directory if start_new_model is True
    if self.is_master and self.start_new_model and exists(self.train_dir):
      utils.remove_training_directory(self.train_dir)
    if self.is_master and self.start_new_model:
      mkdir(self.train_dir)

    if self.params.torch_random_seed is not None:
      random.seed(self.params.torch_random_seed)
      torch.manual_seed(self.params.torch_random_seed)
      cudnn.deterministic = True
      warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if self.params.cudnn_benchmark:
      cudnn.benchmark = True

    self._run_training()

  def _run_training(self):

    # if start_new_model is True, global_step = 0
    # else we get global step from checkpoint
    if self.start_new_model:
      start_epoch = 0
      global_step = 0
    else:
      start_epoch = self.checkpoint['epoch']
      global_step = self.checkpoint['global_step']

    data_loader, sampler = self.reader.load_dataset()
    if sampler is not None:
      assert sampler.num_replicas == self.world_size

    batch_size = self.batch_size
    if self.is_distributed:
      n_files = sampler.num_samples
    else:
      n_files = self.reader.n_train_files

    if self.local_rank == 0:
      logging.info("Number of files on worker: {}".format(n_files))
      logging.info("Start training")

    profile_enabled = False
    for epoch_id in range(start_epoch, self.params.num_epochs):
      if self.is_distributed:
        sampler.set_epoch(epoch_id)
      for n_batch, data in enumerate(data_loader):
        epoch = (int(global_step) * batch_size) / n_files
        with torch.autograd.profiler.profile(
            enabled=profile_enabled, use_cuda=True) as prof:
          self._training(data, epoch, global_step)
        if profile_enabled:
          logging.info(prof.key_averages().table(sort_by="self_cpu_time_total"))
        self._save_ckpt(global_step, epoch_id)
        global_step += 1

      self.scheduler.step()
    self._save_ckpt(global_step, epoch_id, final=True)
    logging.info("Done training -- epoch limit reached.")


  def _process_gradient(self, step):
    if self.params.gradient_clip_by_norm:
      if step == 0 and self.local_rank == 0:
        logging.info("Clipping Gradient by norm: {}".format(
          self.params.gradient_clip_by_norm))
      torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), self.params.gradient_clip_by_norm)
    elif self.params.gradient_clip_by_value:
      if step == 0 and self.local_rank == 0:
        logging.info("Clipping Gradient by value: {}".format(
          self.params.gradient_clip_by_value))
      torch.nn.utils.clip_grad_value_(
        self.model.parameters(), self.params.gradient_clip_by_value)


  def _training(self, data, epoch, step):

    batch_start_time = time.time()
    n_classes = self.reader.n_classes
    inputs, labels = data
    inputs = inputs.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    # generate adversarial attacks
    if self.params.adversarial_training:
      if inputs.min() < 0 or inputs.max() > 1:
        raise ValueError('Input values should be in the [0, 1] range.')
      inputs = self.attack.perturb(inputs)

    outputs = self.model(inputs)
    loss = self.criterion(outputs, labels)

    self.optimizer.zero_grad()
    loss.backward()
    self._process_gradient(step)
    self.optimizer.step()

    # update ema
    if self.ema is not None:
      self.ema(self.model, step)

    seconds_per_batch = time.time() - batch_start_time
    examples_per_second = self.batch_size / seconds_per_batch
    examples_per_second *= self.world_size
    
    if step == 10 and self.is_master:
      n_imgs_to_process = self.reader.n_train_files * self.params.num_epochs
      total_seconds = n_imgs_to_process / examples_per_second
      n_days = total_seconds // 86400
      n_hours = (total_seconds % 86400) / 3600
      logging.info(
        'Approximated training time: {:.0f} days and {:.1f} hours'.format(
          n_days, n_hours))

    local_rank = self.local_rank
    to_print = step % self.params.frequency_log_steps == 0
    if (to_print and local_rank == 0) or (step == 1 and local_rank == 0):
      lr = self.optimizer.param_groups[0]['lr']
      self.message.add("epoch", epoch, format="4.2f")
      self.message.add("step", step, width=5, format=".0f")
      self.message.add("lr", lr, format=".6f")
      self.message.add("loss", loss, format=".4f")
      self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
      logging.info(self.message.get_message())




