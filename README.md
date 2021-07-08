# Stable ResNet Library

## Setup library

```
git clone https://github.com/araujoalexandre/stable_resnet.git

# setup the paths in .bash_profile

eval $(idrenv -d yxj)
export slurm_account=yxj@gpu
export PROJECTDIR=~/stable_resnet
export DATADIR=$SCRATCH/data
export DATADIR=$DATADIR:$WORK/data
export DATADIR=$DATADIR:$DSDIR
export WORKDIR=$SCRATCH
```

## Run train and eval

The config.yaml file should be in the config folder
```
./sub/submit.py --config=cifar10
```
The outout should be:
```
Submitted batch job 123456 123457
Folder '2021-XX-XX_XX.XX.XX_XXXX' created.
```

Two jobs have been submitted: 123456 for training and 123457 for the evaluation as a dependancy of the training job.

The folders '2021-XX-XX_XX.XX.XX_XXXX' and '2021-XX-XX_XX.XX.XX_XXXX_logs' have been cerated in the directory WORKDIR setup above.
'2021-XX-XX_XX.XX.XX_XXXX' for the checkpoints and '2021-XX-XX_XX.XX.XX_XXXX_logs' for the logs.

It's possible to run a training without the eval job with:
```
./sub/submit.py --config=resflow --no_eval
```

To run the eval job after training manualy:
```
./sub/submit.py --mode=eval --train_dir=2021-XX-XX_XX.XX.XX_XXXX
```

To run the training on several machines:
```
./sub/submit.py --config=resflow --nodes 10
```

Checking, and canceling jobs:
```
squeue -u userid
scancel 123456 123457
```

## Grid Search

To launch a series of experiment with different parameters, it possible to use the config file as a template and populate it with values. 

Example of config file as template:
```
default: &DEFAULT
  train_batch_size:           {batch_size}
  num_epochs:                 {epochs}
  start_new_model:            True
  num_gpu:                    2
...
```

You can populate the values with the "params" parameter:
```
./sub/submit.py --config=resflow --grid_search="epochs:5,10,15;batch_size:32,64,128}"
```



