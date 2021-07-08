
import logging
import socket
import pprint
import argparse
from os.path import join
import utils
import torch


def main(args):
  
  config_path = join('{}_logs'.format(args.train_dir), args.config_file)
  config_name = args.config_name
  override_params = args.override_params

  params = utils.load_params(
    config_path, config_name, override_params=override_params)
  params.train_dir = args.train_dir
  params.data_dir = args.data_dir
  params.num_gpus = args.n_gpus
  params.start_new_model = False

  # Setup logging & log the version.
  utils.setup_logging(params.logging_verbosity)

  # print self.params parameters
  pp = pprint.PrettyPrinter(indent=2, compact=True)
  logging.info(pp.pformat(params.values()))
  logging.info("Pytorch version: {}.".format(torch.__version__))
  logging.info("Hostname: {}.".format(socket.gethostname()))

  core_eval = __import__('core.{}'.format(config_name), fromlist=[''])
  evaluate = core_eval.Evaluator(params)
  evaluate.run()

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Main eval script.')

  parser.add_argument("--config_file", type=str, default="config.yaml",
                      help="Name of the yaml config file.")
  parser.add_argument("--config_name", type=str, default="eval",
                      help="Define the execution mode.")
  parser.add_argument("--train_dir", type=str, required=True,
                      help="Name of the training directory")
  parser.add_argument("--data_dir", type=str, required=True,
                      help="Name of the data directory")
  parser.add_argument("--n_gpus", type=int, default=4,
                      help="Number of GPUs to use.")
  parser.add_argument("--override_params", type=str,
                      help="Parameters to override.")

  args = parser.parse_args()
  main(args)

