
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
# import multiprocessing
import torch.multiprocessing as multiprocessing


# np.random.seed(0)
# torch.manual_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--config', type=str, help="./configs/train_base.yaml")
args = parser.parse_args()


def main(config):
    pprint.pprint(config)

    if os.environ.get('DRYRUN', '0') == '1':
        config['wandb']['mode'] = 'disabled'

    wandb_run = wandb.init(
        **config['wandb'],
        config=config,
    )
    wandb_run.save(args.config)
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
    # Load YAML config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    config = yaml.safe_load(
    # r"""
    # method: vanilla
    
    # data:
    #   dataset: noisy_cifar10
    #   noise_type: symmetric
    #   noise_rate: 0.5
    #   download: true
    
    # model:
    #   architecture: resnet18
    #   num_classes: 10
    
    # wandb:
    #   mode: online # "disabled" or "online"
    #   entity: hyounguk-shon
    #   project: noisy-label
    #   name: CIFAR10-CE
    #   save_code: true
    
    # trainer:
    #   optimizer: sgd
    #   init_lr: 0.1
    #   momentum: 0.9
    #   weight_decay: 1.0e-4
    #   lr_scheduler: multistep
    #   max_epoch: 200
    #   num_workers: 4
    #   batch_size: 128
    #   save_model: true
    #   loss_fn: cross_entropy
    #   alpha: 0.5
    #   teacher_aug: randomcrop
    #   student_aug: gaussianblur
    #   distill_loss_fn: kl_div
    #   temperature: 1.0
    #   enable_amp: false
    # """
    """
    method: vanilla

    data:
        dataset: cifar10
        download: true

    model:
        architecture: resnet18
        num_classes: 10

    wandb:
        mode: online # "disabled" or "online"
        entity: hyounguk-shon
        project: noisy-label
        name: CIFAR10-CE
        save_code: true

    trainer:
        optimizer: sgd
        init_lr: 1.0e-1
        momentum: 0.9
        weight_decay: 1.0e-4
        lr_scheduler: multistep
        max_epoch: 200
        num_workers: 2
        batch_size: 128
        save_model: true
        loss_fn: cross_entropy
        aug: none
        enable_amp: false

    """
    )

    # for teacher_aug in ['randomcrop', 'gaussianblur', 'rotate', 'colorjitter']:
    #     for student_aug in ['randomcrop', 'gaussianblur', 'rotate', 'colorjitter']:
    #         config['trainer']['teacher_aug'] = teacher_aug
    #         config['trainer']['student_aug'] = student_aug
    #         main(config)

    # for aug in ['randomcrop', 'gaussianblur', 'rotate', 'colorjitter']:
    #     config['trainer']['aug'] = aug
    #     main(config)


    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    # run a list of configs in parallel
    with ProcessPoolExecutor(max_workers=2) as pool:
        for aug in ['randomcrop', 'gaussianblur', 'rotate', 'colorjitter']:
            config['trainer']['aug'] = aug
            pool.submit(main, deepcopy(config))
        pool.shutdown(wait=True)


