
from . import resnet_model_cifar
from . import resnet_model_stable
from . import resnet_model_imagenet

_model_name_to_imagenet_model = {
    'resnet': resnet_model_imagenet.resnet,
}

_model_name_to_cifar_model = {
    'resnet': resnet_model_cifar.ResNet,
    'resnet_stable' : resnet_model_stable.ResNet
}

def _get_model_map(dataset_name):
  """Get name to model map for specified dataset."""
  if dataset_name in ('cifar10', 'cifar100'):
    return _model_name_to_cifar_model
  elif dataset_name == 'mnist':
    return _model_name_to_mnist_model
  elif dataset_name in ('imagenet'):
    return _model_name_to_imagenet_model
  else:
    raise ValueError('Invalid dataset name: {}'.format(dataset_name))


def get_model_config(model_name, dataset_name, params, nclass, is_training):
  """Map model name to model network configuration."""
  model_map = _get_model_map(dataset_name)
  if model_name not in model_map:
    raise ValueError("Invalid model name '{}' for dataset '{}'".format(
                     model_name, dataset_name))
  else:
    return model_map[model_name](params, nclass, is_training)



