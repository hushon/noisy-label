from .cifar import CIFAR10, CIFAR100, NoisyCIFAR10, NoisyCIFAR100, CIFAR10N, CIFAR100N, NoisyCIFAR3
from .food101 import Food101
from .animal import Animal10N
from .clothing import Clothing1M
import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2
from typing import Tuple
from torch.utils.data import Dataset


def get_dataset(**kwargs) -> Tuple[Dataset, Dataset]:
    data_root = "./data"
    dataset_name = kwargs.pop('dataset')
    CIFAR10_MEAN_STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    CIFAR100_MEAN_STD = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    IMAGENET_MEAN_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    match dataset_name:
        case "noisy_cifar10":
            transform_train = transforms.Compose([
                transforms_v2.RandomCrop(32, padding=[4,]),
                transforms_v2.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
            ])
            train_dataset = NoisyCIFAR10(data_root, transform=transform_train, **kwargs)
            test_dataset = CIFAR10(data_root, train=False, transform=transform_test)

        case "noisy_cifar100":
            transform_train = transforms.Compose([
                transforms_v2.RandomCrop(32, padding=[4,]),
                transforms_v2.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*CIFAR100_MEAN_STD, inplace=True),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
            ])
            train_dataset = NoisyCIFAR100(data_root, transform=transform_train, **kwargs)
            test_dataset = CIFAR100(data_root, train=False, transform=transform_test)

        case "noisy_cifar3":
            transform_train = transforms.Compose([
                transforms_v2.RandomCrop(32, padding=[4,]),
                transforms_v2.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
            ])
            train_dataset = NoisyCIFAR3(data_root, train=True, transform=transform_train, **kwargs)
            test_dataset = NoisyCIFAR3(data_root, train=False, transform=transform_test, **kwargs)

        case "clothing1m":
            transform_train = transforms.Compose([
                transforms_v2.RandomResizedCrop(224),
                transforms_v2.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*IMAGENET_MEAN_STD, inplace=True),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(*IMAGENET_MEAN_STD, inplace=True),
            ])
            train_dataset = Clothing1M(data_root, split='noisy_train', transform=transform_train, **kwargs)
            test_dataset = Clothing1M(data_root, split='clean_test', transform=transform_test, **kwargs)

        case _:
            raise NotImplementedError

    return train_dataset, test_dataset
