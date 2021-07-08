
import argparse
import utils
from core.trainer import Trainer

def main(args):

  # the function load_params will load the yaml config file and 
  # override parameters if necessary
  params = utils.load_params(args.config_file, args.config_name)

  params.train_dir = args.train_dir
  params.data_dir = args.data_dir
  params.start_new_model = args.start_new_model
  params.num_gpus = args.n_gpus
  params.job_name = args.job_name
  params.local_rank = args.local_rank
  params.ps_hosts = args.ps_hosts
  params.worker_hosts = args.worker_hosts
  params.master_host = args.master_host
  params.master_port = args.master_port
  params.task_index = args.task_index

  trainer = Trainer(params)
  trainer.run()


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Main train script.')

  parser.add_argument("--config_file", type=str,
                      help="Name of the yaml config file.")
  parser.add_argument("--config_name", type=str, default="train",
                      help="Define the execution mode.")
  parser.add_argument("--train_dir", type=str, required=True,
                      help="Name of the training directory")
  parser.add_argument("--data_dir", type=str, required=True,
                      help="Name of the data directory")
  parser.add_argument("--start-new-model", action="store_true", default=False,
                      help="Start training a new model or restart an existing one.")
  parser.add_argument("--job_name", type=str, choices=('ps', 'worker', ''),
                      help="Type of job 'ps', 'worker', ''.")
  parser.add_argument("--n_gpus", type=int, default=4, 
                      help="Number of GPUs to use.")
  parser.add_argument("--local_rank", type=int, default=0, 
                      help="Define local rank of the worker.")
  parser.add_argument("--ps_hosts", type=str,
                      help="Comma-separated list of target hosts.")
  parser.add_argument("--worker_hosts", type=str, 
                      help="Comma-separated list of target hosts.")
  parser.add_argument("--master_host", type=str,
                      help="ip/hostname of the master.")
  parser.add_argument("--master_port", type=str,
                      help="port of the master.")
  parser.add_argument("--task_index", type=int, default=0,
                      help="Index of task within the job")

  args = parser.parse_args()
  main(args)



