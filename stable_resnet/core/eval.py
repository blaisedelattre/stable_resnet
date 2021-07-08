
import json
import time
import os
import re
import socket
import pprint
import logging
from os.path import join, exists, basename

import utils
from dump_files import DumpFiles
from .models import model_config
from .dataset.readers import readers_config
from .randomized_smoothing import Smooth

import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.backends.cudnn as cudnn


class Evaluator:
  """Evaluate a Pytorch Model."""

  def __init__(self, params):

    self.params = params

    self.train_dir = self.params.train_dir
    self.logs_dir = "{}_logs".format(self.train_dir)
    if self.train_dir is None:
      raise ValueError('Trained model directory not specified')
    self.num_gpus = self.params.num_gpus

    # create a mesage builder for logging
    self.message = utils.MessageBuilder()

    if self.params.cudnn_benchmark:
      cudnn.benchmark = True

    if self.params.num_gpus:
      self.batch_size = self.params.batch_size * self.num_gpus
    else:
      self.batch_size = self.params.batch_size

    if not self.params.data_pattern:
      raise IOError("'data_pattern' was not specified. "
        "Nothing to evaluate.")

    # load reader
    self.reader = readers_config[self.params.dataset](
      self.params, self.batch_size, self.num_gpus, is_training=False)

    # load model
    self.model = model_config.get_model_config(
        self.params.model, self.params.dataset, self.params,
        self.reader.n_classes, is_training=False)
    # add normalization as first layer of model
    if self.params.add_normalization:
      normalize_layer = self.reader.get_normalize_layer()
      self.model = torch.nn.Sequential(normalize_layer, self.model)
    self.model = torch.nn.DataParallel(self.model)
    self.model = self.model.cuda()

    # init loss
    self.criterion = torch.nn.CrossEntropyLoss().cuda()

    # save files for analysis
    if self.params.dump_files:
      assert self.params.eval_under_attack, \
          "dumping files only available when under attack"
      self.dump = DumpFiles(params)

    # eval under attack
    if self.params.eval_under_attack:
      attack_params = self.params.attack_params
      self.attack = utils.get_attack(
                      self.model,
                      self.reader.n_classes,
                      self.params.attack_method,
                      attack_params)

    if self.params.additive_noise or self.params.adaptive_noise:
      # define Smooth classifier
      dim = np.product(self.reader.img_size[1:])
      self.smooth_model = Smooth(
        self.model, self.params, self.reader.n_classes, dim)

  def run(self):
    """Run evaluation of model or eval under attack"""
    logging.info("Building evaluation graph")
    if not self.params.eval_under_attack:
      self._run_eval()
      logging.info("Done evaluation -- number of eval reached.")
    else:
      self._run_under_attack()
      logging.info('Evalution under attack done.')


  def load_ckpt(self, path):
    checkpoint = torch.load(path)
    global_step = checkpoint.get('global_step', 250)
    epoch = checkpoint.get('epoch', 1)
    if 'ema' in checkpoint.keys():
      logging.info("Loading ema state dict.")
      self.model.load_state_dict(checkpoint['ema'])
    else:
      if 'model_state_dict' not in checkpoint.keys():
        # if model_state_dict is not in checkpoint.keys() the 
        # checkpoint must be a pre-trained model
        # we should fix the problem of 'module'
        new_state_dict = {}
        for k, v in checkpoint.items():
          new_name = 'module.' + k
          new_state_dict[new_name] = v
        self.model.load_state_dict(new_state_dict)
      else:
        self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.eval()
    return global_step, epoch


  def _run_eval(self):
    # those variables are updated in eval_loop
    self.best_global_step = None
    self.best_accuracy = None
    ckpts = utils.get_list_checkpoints(self.train_dir)
    for ckpt in ckpts[::-1]:
      # remove first checkpoint model.ckpt-0
      if 'model.ckpt-0' in ckpt: continue
      logging.info(
        "Loading checkpoint for eval: {}".format(basename(ckpt)))
      global_step, epoch = self.load_ckpt(ckpt)
      self.eval_loop(global_step, epoch)

    if self.best_global_step is not None and self.best_accuracy is not None:
      path = join(self.logs_dir, "best_accuracy.txt")
      with open(path, 'a') as f:
        f.write("{}\t{:.4f}\n".format(
          self.best_global_step, self.best_accuracy))


  def _run_under_attack(self):
    # normal evaluation has already been done
    # we get the best checkpoint of the model
    best_checkpoint, global_step = utils.get_best_checkpoint(self.logs_dir)
    logging.info("Loading '{}'".format(best_checkpoint.split('/')[-1]))
    global_step, epoch = self.load_ckpt(best_checkpoint)
    self.eval_attack(global_step, epoch)

    # save results
    path = join(self.logs_dir, "attacks_score.txt")
    with open(path, 'a') as f:
      f.write("{}\n".format(self.attack.__class__.__name__))
      if self.params.eot:
        f.write("eot sample {}, {}\n".format(self.eot_samples,
                                       json.dumps(self.params.attack_params)))
      else:
        f.write("{}\n".format(json.dumps(self.params.attack_params)))
      f.write("{:.5f}\n\n".format(self.best_accuracy))


  def eval_loop(self, global_step, epoch):
    """Run the evaluation loop once."""

    running_inputs = 0
    running_accuracy = 0
    running_accuracy_smooth = 0
    # running_loss = 0
    # running_loss_smooth = 0
    data_loader, _ = self.reader.load_dataset()
    for batch_n, data in enumerate(data_loader):

      with torch.no_grad():
        batch_start_time = time.time()
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        idx1 = list(range(inputs.shape[0]))

        outputs = self.model(inputs)
        # loss = self.criterion(outputs, labels)
        predicted = outputs.argmax(axis=1)
        running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
        running_inputs += inputs.size(0)
        # running_loss += loss.cpu().numpy()
        accuracy = running_accuracy / running_inputs
        # loss = running_loss / (batch_n + 1)

        outputs_smooth = self.smooth_model(inputs)
        # loss_smooth = (-torch.log(outputs_smooth)[idx1, labels]).mean()
        predicted_smooth = outputs_smooth.argmax(axis=1)
        running_accuracy_smooth += predicted_smooth.eq(labels.data).cpu().sum().numpy()
        # running_loss_smooth += loss_smooth.cpu().numpy()
        accuracy_smooth = running_accuracy_smooth / running_inputs
        # loss_smooth = running_loss_smooth / (batch_n + 1)

      seconds_per_batch = time.time() - batch_start_time
      examples_per_second = inputs.size(0) / seconds_per_batch

      self.message.add('epoch', epoch)
      self.message.add('step', global_step)
      # self.message.add('accuracy', accuracy, format='.5f')
      # self.message.add('loss', loss, format='.5f')
      self.message.add('accuracy', [accuracy, accuracy_smooth], format='.5f')
      # self.message.add('loss', [loss, loss_smooth], format='.5f')
      self.message.add('imgs/sec', examples_per_second, format='.0f')
      logging.info(self.message.get_message())

    if self.best_accuracy is None or self.best_accuracy < accuracy_smooth:
      self.best_global_step = global_step
      self.best_accuracy = accuracy_smooth
    self.message.add('--> epoch', epoch)
    self.message.add('step', global_step)
    self.message.add('accuracy', accuracy_smooth, format='.5f')
    # self.message.add('loss', loss, format='.5f')
    logging.info(self.message.get_message())
    logging.info("Done with batched inference.")
    return

  def eval_attack(self, global_step, epoch):
    """Run evaluation under attack."""

    running_accuracy = 0
    running_inputs = 0
    # running_loss = 0
    data_loader, _ = self.reader.load_dataset()
    for batch_n, data in enumerate(data_loader):

      batch_start_time = time.time()
      inputs, labels = data
      inputs, labels = inputs.cuda(), labels.cuda()

      # craft attack
      if inputs.min() < 0 or inputs.max() > 1:
        raise ValueError('Input values should be in the [0, 1] range.')
      inputs_adv = self.attack.perturb(inputs, labels)

      # predict
      outputs = self.model_smooth(inputs)
      outputs_adv = self.model_smooth(inputs_adv)

      # loss = self.criterion(outputs_adv, labels)
      _, predicted = torch.max(outputs_adv.data, 1)
      seconds_per_batch = time.time() - batch_start_time
      examples_per_second = inputs.size(0) / seconds_per_batch

      running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
      running_inputs += inputs.size(0)
      # running_loss += loss.cpu().detach().numpy()
      accuracy = running_accuracy / running_inputs
      # loss = running_loss / (batch_n + 1)

      if self.params.dump_files:
        results = {
          'images': inputs.cpu().numpy(),
          'images_adv': inputs_adv.cpu().numpy(),
          'predictions': outputs.detach().cpu().numpy(),
          'predictions_adv': outputs_adv.detach().cpu().numpy()
        }
        self.dump.files(results)

      self.message.add('', socket.gethostname())
      self.message.add('accuracy', accuracy, format='.5f')
      # self.message.add('loss', loss, format='.5f')
      self.message.add('imgs/sec', examples_per_second, format='.0f')
      logging.info(self.message.get_message())

    self.best_global_step = global_step
    self.best_accuracy = accuracy
    self.message.add('', socket.gethostname())
    self.message.add('accuracy', accuracy, format='.5f')
    # self.message.add('loss', loss, format='.5f')
    logging.info(self.message.get_message())
    logging.info("Done with batched inference under attack.")
    return

