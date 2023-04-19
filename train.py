
import argparse
import os
import torch
import numpy as np
import random
import wandb
from models import resnet
import yaml


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--config', type=str, default='./configs/train_base.yml')
args = parser.parse_args()

def get_dataset(dataset, noise_rate, noise_type):
    if dataset == "noisy_cifar10":
        return NoisyCIFAR10(noise_rate=noise_rate, noise_type=noise_type)
    elif dataset == "noisy_cifar100":
        return NoisyCIFAR100(noise_rate=noise_rate, noise_type=noise_type)


def get_model(model_name, num_classes):
    if model_name == "resnet18":
        return resnet.resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        return resnet.resnet34(num_classes=num_classes)
    elif model_name == "resnet50":
        return resnet.resnet50(num_classes=num_classes)
    elif model_name == "resnet101":
        return resnet.resnet101(num_classes=num_classes)
    elif model_name == "resnet152":
        return resnet.resnet152(num_classes=num_classes)
    else:
        raise NotImplementedError


def main():
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    wandb.init(
        **config['wandb'],
        config=config,
    )

    device = torch.device("cuda")

    train_loader, test_loader = get_dataloader(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        )

    model = get_model(
        model_name=args.model,
        num_classes=10,
    ).to(device)


    trainer = Trainer(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    model=model,
                    **config['trainer'],
                    )

    trainer.fit()

    wandb.finish()

if __name__ == '__main__':
    main()