from .cifar import CIFAR10, CIFAR100, NoisyCIFAR10, NoisyCIFAR100, CIFAR10N, CIFAR100N, NoisyCIFAR3
from .food101 import Food101
from .animal import Animal10N
from torchvision import transforms
from typing import Tuple
import torch.utils.data as data


def get_dataset(**kwargs) -> Tuple[data.Dataset, data.Dataset]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_root = "./data"
    dataset = kwargs.pop('dataset')
    if dataset == "noisy_cifar10":
        train = NoisyCIFAR10(data_root, transform=transform_train, **kwargs)
        test = CIFAR10(data_root, train=False, transform=transform_test)
    elif dataset == "noisy_cifar100":
        train = NoisyCIFAR100(data_root, transform=transform_train, **kwargs)
        test = CIFAR100(data_root, train=False, transform=transform_test)
    elif dataset == "noisy_cifar3":
        train = NoisyCIFAR3(data_root, train=True, transform=transform_train, **kwargs)
        test = NoisyCIFAR3(data_root, train=False, transform=transform_test, **kwargs)
    else:
        raise NotImplementedError
    return train, test