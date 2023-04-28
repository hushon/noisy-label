from .cifar import CIFAR10, CIFAR100, NoisyCIFAR10, NoisyCIFAR100, CIFAR10N, CIFAR100N
from .food101 import Food101
from animal import Animal10N
from torchvision import transforms
from typing import Tuple
import torch.utils.data as data


def get_dataset(dataset, noise_rate, noise_type, random_seed) -> Tuple[data.Dataset, data.Dataset]:
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
    if dataset == "noisy_cifar10":
        train = NoisyCIFAR10("./data", download=True, transform=transform_train, noise_rate=noise_rate, noise_type=noise_type, random_seed=random_seed)
        test = CIFAR10("./data", download=True, train=False, transform=transform_test)
    elif dataset == "noisy_cifar100":
        train = NoisyCIFAR100("./data", download=True, transform=transform_train, noise_rate=noise_rate, noise_type=noise_type, random_seed=random_seed)
        test = CIFAR100("./data", download=True, train=False, transform=transform_test)
    else:
        raise NotImplementedError
    return train, test