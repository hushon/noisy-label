
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
from torchvision import transforms
from datasets import get_dataset
from models import get_model


np.random.seed(0)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--config', type=str, required=True, help="./configs/train_base.yaml")
args = parser.parse_args()


def main():
    # Load YAML config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    pprint.pprint(config)

    wandb_run = wandb.init(
        **config['wandb'],
        config=config,
    )
    wandb_run.save(args.config)
    wandb_run.log_code()

    train_dataset, test_dataset = get_dataset(**config["data"])

    # model = get_model(**config["model"])
    import torchvision.models
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(2048, 14)

    trainer = Trainer(
                    model=model,
                    config=config['trainer'],
                    wandb_run=wandb_run,
                    )

    trainer.fit(train_dataset, test_dataset)


    # wandb_run.alert(
    #     title="Training finished",
    #     text="this is a test message",
    #     level=wandb.AlertLevel.INFO,
    # )
    wandb_run.finish()

if __name__ == '__main__':
    main()