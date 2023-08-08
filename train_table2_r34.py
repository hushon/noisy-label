
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def main(config):
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False))

    if os.environ.get('DRYRUN', '0') == '1':
        config['wandb']['mode'] = 'disabled'

    wandb_run = wandb.init(
        **config['wandb'],
        config=config,
    )
    wandb_run.log_code()

    # for multiprocessing
    if multiprocessing.current_process().name == 'MainProcess':
        device = "cuda:0"
    else:
        worker_id = multiprocessing.current_process().name
        print(f"{worker_id=}")
        worker_id = int(worker_id.split('-')[1]) - 1
        device = f"cuda:{worker_id}"

    model = get_model(**config["model"])
    trainer = Trainer(
                    model=model,
                    config=config['trainer'],
                    wandb_run=wandb_run,
                    device=device,
                    )

    train_dataset, test_dataset = get_dataset(**config["data"])


    match config['method']:
        case 'vanilla':
            trainer.fit(train_dataset, test_dataset)
        case 'nrd':
            trainer.fit_nrosd(train_dataset, test_dataset)
        case 'nrd_hardlabel':
            trainer.fit_nrosd_hardlabel(train_dataset, test_dataset)
        case _:
            raise NotImplementedError

    wandb_run.finish()


if __name__ == '__main__':
    config_files = [
        './configs/cifar10n/cifar10n_ce.yaml',
        './configs/cifar10n/cifar10n_ce_nrosd.yaml',
        './configs/cifar10n/cifar10n_mae.yaml',
        './configs/cifar10n/cifar10n_mae_nrosd.yaml',
        './configs/cifar10n/cifar10n_gce.yaml',
        './configs/cifar10n/cifar10n_sce.yaml',
        './configs/cifar100n/cifar100n_ce.yaml',
        './configs/cifar100n/cifar100n_ce_nrosd.yaml',
        './configs/cifar100n/cifar100n_mae.yaml',
        './configs/cifar100n/cifar100n_mae_nrosd.yaml',
        './configs/cifar100n/cifar100n_gce.yaml',
        './configs/cifar100n/cifar100n_sce.yaml'
    ]


    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    pool = ProcessPoolExecutor(max_workers=max_workers)
    for path in config_files:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        pool.submit(main, deepcopy(config))

    pool.shutdown()



