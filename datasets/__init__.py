from .cifar import CIFAR10, CIFAR100, NoisyCIFAR10, NoisyCIFAR100, CIFAR10N, CIFAR100N, NoisyCIFAR3, OldNoisyCIFAR10, OldNoisyCIFAR100
from .food101 import Food101
from .animal import Animal10N
from .clothing import Clothing1M
from .webvision import WebVisionV1
import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from models import Normalize2D


CIFAR10_MEAN_STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CIFAR100_MEAN_STD = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
IMAGENET_MEAN_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def get_dataset(**kwargs) -> Tuple[Dataset, Dataset]:
    if 'root' in kwargs:
        data_root = kwargs.pop('root')
    else:
        data_root = "/dev/shm/data/"
    if 'random_seed' in kwargs:
        random_seed = kwargs.pop('random_seed')
    dataset_name = kwargs.pop('dataset')

    match dataset_name:
        case "noisy_cifar10":
            train_dataset = NoisyCIFAR10(data_root, download=True, random_seed=random_seed, **kwargs)
            test_dataset = CIFAR10(data_root, train=False, download=True)
        case "noisy_cifar100":
            train_dataset = NoisyCIFAR100(data_root, download=True, random_seed=random_seed, **kwargs)
            test_dataset = CIFAR100(data_root, train=False, download=True)
        case "old_noisy_cifar10":
            train_dataset = OldNoisyCIFAR10(data_root, download=True, random_seed=random_seed, **kwargs)
            test_dataset = CIFAR10(data_root, train=False, download=True)
        case "old_noisy_cifar100":
            train_dataset = OldNoisyCIFAR100(data_root, download=True, random_seed=random_seed, **kwargs)
            test_dataset = CIFAR100(data_root, train=False, download=True)
        case "noisy_cifar3":
            train_dataset = NoisyCIFAR3(data_root, train=True, download=True, random_seed=random_seed, **kwargs)
            test_dataset = NoisyCIFAR3(data_root, train=False, download=True, random_seed=random_seed, **kwargs)
        case "clothing1m":
            train_dataset = Clothing1M(data_root, split='noisy_train', **kwargs)
            test_dataset = Clothing1M(data_root, split='clean_test', **kwargs)
        case "cifar10n":
            train_dataset = CIFAR10N(data_root, train=True, download=True, **kwargs)
            test_dataset = CIFAR10(data_root, train=False, download=True)
        case "cifar100n":
            train_dataset = CIFAR100N(data_root, train=True, download=True, **kwargs)
            test_dataset = CIFAR100(data_root, train=False, download=True)
        case "animal10n":
            train_dataset = Animal10N(data_root, train=True, download=True, **kwargs)
            test_dataset = Animal10N(data_root, train=False, download=True)
        case "webvision":
            train_dataset = WebVisionV1(data_root, split='train', num_classes=50, **kwargs)
            test_dataset = WebVisionV1(data_root, split='val', num_classes=50)
        case "cifar10":
            train_dataset = CIFAR10(data_root, train=True, download=True, **kwargs)
            test_dataset = CIFAR10(data_root, train=False, download=True)
        case "cifar100":
            train_dataset = CIFAR100(data_root, train=True, download=True, **kwargs)
            test_dataset = CIFAR100(data_root, train=False, download=True)
        case _:
            raise NotImplementedError(dataset_name)
    return train_dataset, test_dataset


def get_transform(op_name: str, dataset: Dataset) -> nn.Module:
    dataset_type = type(dataset)
    match op_name:
        case "none":
            if dataset_type in [CIFAR10, NoisyCIFAR10, NoisyCIFAR3, CIFAR10N, CIFAR100, NoisyCIFAR100, CIFAR100N, OldNoisyCIFAR10, OldNoisyCIFAR100]:
                return nn.Sequential(
                    transforms_v2.ToImageTensor(),
                )
            elif dataset_type in [Clothing1M, WebVisionV1]:
                return nn.Sequential(
                    transforms_v2.Resize(256, antialias=True),
                    transforms_v2.CenterCrop(224),
                    transforms_v2.ToImageTensor(),
                )
            elif dataset_type in [Animal10N]:
                return nn.Sequential(
                    transforms_v2.ToImageTensor(),
                    # transforms_v2.Normalize(*IMAGENET_MEAN_STD, inplace=True),
                )
            else:
                raise NotImplementedError(dataset_type)
        case "randomcrop":
            if dataset_type in [CIFAR10, NoisyCIFAR10, NoisyCIFAR3, CIFAR10N, CIFAR100, NoisyCIFAR100, CIFAR100N, OldNoisyCIFAR10, OldNoisyCIFAR100]:
                return nn.Sequential(
                    transforms_v2.RandomCrop(32, padding=4),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.ToImageTensor(),
                )
            elif dataset_type in [Clothing1M, WebVisionV1]:
                return nn.Sequential(
                    transforms_v2.Resize(256, antialias=True),
                    transforms_v2.RandomCrop(224),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.ToImageTensor(),
                )
            elif dataset_type in [Animal10N]:
                return nn.Sequential(
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms_v2.ToImageTensor(),
                    # transforms.Normalize(*IMAGENET_MEAN_STD, inplace=True),
                )
            else:
                raise NotImplementedError(dataset_type)
        case "autoaugment":
            if dataset_type in [CIFAR10, NoisyCIFAR10, NoisyCIFAR3, CIFAR10N, CIFAR100, NoisyCIFAR100, CIFAR100N, OldNoisyCIFAR10, OldNoisyCIFAR100]:
                return nn.Sequential(
                    transforms_v2.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                    transforms_v2.ToImageTensor(),
                )
            elif dataset_type in [Clothing1M, WebVisionV1]:
                return nn.Sequential(
                    transforms_v2.Resize(256, antialias=True),
                    transforms_v2.CenterCrop(224),
                    transforms_v2.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                    transforms_v2.ToImageTensor(),
                )
            else:
                raise NotImplementedError(dataset_type)
        case "gaussianblur":
            if dataset_type in [CIFAR10, NoisyCIFAR10, NoisyCIFAR3, CIFAR10N, CIFAR100, NoisyCIFAR100, CIFAR100N, OldNoisyCIFAR10, OldNoisyCIFAR100]:
                return nn.Sequential(
                    transforms_v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.ToImageTensor(),
                )
            else:
                raise NotImplementedError(dataset_type)
        case "sharpen":
            if dataset_type in [CIFAR10, NoisyCIFAR10, NoisyCIFAR3, CIFAR10N, CIFAR100, NoisyCIFAR100, CIFAR100N, OldNoisyCIFAR10, OldNoisyCIFAR100]:
                return nn.Sequential(
                    transforms_v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.ToImageTensor(),
                )
            else:
                raise NotImplementedError(dataset_type)
        case "rotate":
            if dataset_type in [CIFAR10, NoisyCIFAR10, NoisyCIFAR3, CIFAR10N, CIFAR100, NoisyCIFAR100, CIFAR100N, OldNoisyCIFAR10, OldNoisyCIFAR100]:
                return nn.Sequential(
                    transforms_v2.RandomRotation(degrees=15),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.ToImageTensor(),
                )
            else:
                raise NotImplementedError(dataset_type)
        case "colorjitter":
            if dataset_type in [CIFAR10, NoisyCIFAR10, NoisyCIFAR3, CIFAR10N, CIFAR100, NoisyCIFAR100, CIFAR100N, OldNoisyCIFAR10, OldNoisyCIFAR100]:
                return nn.Sequential(
                    transforms_v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.ToImageTensor(),
                )
            else:
                raise NotImplementedError(dataset_type)
        case _:
            raise NotImplementedError(op_name)


def get_normalization(dataset: Dataset):
    dataset_type = type(dataset)
    if dataset_type in [CIFAR10, NoisyCIFAR10, NoisyCIFAR3, CIFAR10N, OldNoisyCIFAR10]:
        mean, std = CIFAR10_MEAN_STD
    elif dataset_type in [CIFAR100, NoisyCIFAR100, CIFAR100N, OldNoisyCIFAR100]:
        mean, std = CIFAR100_MEAN_STD
    elif dataset_type in [Clothing1M, WebVisionV1]:
        mean, std = IMAGENET_MEAN_STD
    else:
        raise NotImplementedError(dataset_type)
    return nn.Sequential(
        transforms_v2.ConvertDtype(),
        Normalize2D(mean, std),
    )


class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)