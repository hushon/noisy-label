
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
import torch.multiprocessing as multiprocessing


# np.random.seed(0)
# torch.manual_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def main(config):
    # for multiprocessing
    if multiprocessing.current_process().name == 'MainProcess':
        torch.cuda.set_device(0)
    else:
        worker_id = multiprocessing.current_process().name
        print(f"{worker_id=}")
        worker_id = int(worker_id.split('-')[1]) - 1
        torch.cuda.set_device(worker_id)

    wandb_run = wandb.init(
        **config['wandb'],
        config=config,
    )
    wandb_run.log_code()

    train_dataset, test_dataset = get_dataset(**config["data"])

    model = get_model(**config["model"]).cuda()

    trainer = Trainer(
                    model=model,
                    config=config['trainer'],
                    wandb_run=wandb_run,
                    )

    trainer.fit_nrosd(train_dataset, test_dataset)


    # wandb_run.alert(
    #     title="Training finished",
    #     text="this is a test message",
    #     level=wandb.AlertLevel.INFO,
    # )
    wandb_run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('--config', type=str, required=True, help="./configs/train_base.yaml")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    pprint.pprint(config)

    main(config)