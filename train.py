
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

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('high')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def main(config: dict):
    print("=== CONFIG ===")
    print(yaml.dump(config, sort_keys=False))
    print("==============")

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


    # match config['method']:
    #     case 'vanilla':
    #         trainer.fit(train_dataset, test_dataset)
    #     case 'nrd':
    #         trainer.fit_nrosd(train_dataset, test_dataset)
    #     case 'nrd_hardlabel':
    #         trainer.fit_nrosd_hardlabel(train_dataset, test_dataset)
    #     case 'nrd_ema':
    #         trainer.fit_nrosd_ema(train_dataset, test_dataset)
    #     case 'nrd_ema_instance':
    #         trainer.fit_nrosd_ema_instance(train_dataset, test_dataset)
    #     case _:
    #         raise NotImplementedError
    getattr(trainer, config['method'])(train_dataset, test_dataset)

    wandb_run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('config', nargs='?', type=str, default="empty")
    args = parser.parse_args()

    if args.config != 'empty':
        # Load YAML config
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = yaml.safe_load(
        r"""
        method: fit
        
        data:
            dataset: noisy_cifar10
            noise_type: symmetric
            noise_rate: 0.5
            random_seed: 43
        
        model:
            architecture: resnet34
            num_classes: 10
        
        wandb:
            mode: online # "disabled" or "online"
            entity: hyounguk-shon
            project: noisy-label
            name: CIFAR10-CE
            save_code: true
        
        trainer:
            optimizer: sgd
            init_lr: 0.1
            momentum: 0.9
            weight_decay: 1.0e-4
            lr_scheduler: multistep
            max_epoch: 200
            num_workers: 4
            batch_size: 128
            save_model: true
            loss_fn: cross_entropy
            aug: randomcrop
            enable_amp: false
            transform_after_batching: false
        """
        )


    main(config)
