
import argparse
import os
import torch
import numpy as np
import random
import wandb
from models import resnet
import yaml
from trainer import Trainer
import pprint
from torchvision import datasets, transforms
import noisydatasets


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--config', type=str, required=True, help="./configs/train_base.yml")
args = parser.parse_args()

def get_dataset(dataset, noise_rate, noise_type):
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
        train = noisydatasets.NoisyCIFAR10("./data", download=True, transform=transform_train, noise_rate=noise_rate, noise_type=noise_type)
        test = datasets.CIFAR10("./data", download=True, train=False, transform=transform_test)
    elif dataset == "noisy_cifar100":
        train = noisydatasets.NoisyCIFAR100("./data", download=True, transform=transform_train, noise_rate=noise_rate, noise_type=noise_type)
        test = datasets.CIFAR100("./data", download=True, train=False, transform=transform_test)
    else:
        raise NotImplementedError
    return train, test


def get_model(architecture, num_classes):
    if architecture == "resnet18":
        return resnet.resnet18(pretrained=False, in_channels=3, num_classes=num_classes)
    elif architecture == "resnet34":
        return resnet.resnet34(pretrained=False, in_channels=3, num_classes=num_classes)
    elif architecture == "resnet50":
        return resnet.resnet50(pretrained=False, in_channels=3, num_classes=num_classes)
    elif architecture == "resnet101":
        return resnet.resnet101(pretrained=False, in_channels=3, num_classes=num_classes)
    elif architecture == "resnet152":
        return resnet.resnet152(pretrained=False, in_channels=3, num_classes=num_classes)
    else:
        raise NotImplementedError


def main():
    # Load YAML config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    pprint.pprint(config)

    wandb_run = wandb.init(
        **config['wandb'],
        config=config,
    )


    train_dataset, test_dataset = get_dataset(**config["data"])

    model = get_model(**config["model"]).cuda()

    trainer = Trainer(
                    model=model,
                    config=config['trainer'],
                    )

    trainer.fit(train_dataset, test_dataset)

    wandb_run.finish()

if __name__ == '__main__':
    main()