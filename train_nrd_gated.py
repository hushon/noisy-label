
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

    getattr(trainer, config['method'])(train_dataset, test_dataset)

    wandb_run.finish()


if __name__ == '__main__':
    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # ==== CE+ gatedNRD + randomerasing ====
        config = yaml.safe_load(
        r"""
        method: fit_nrosd_multiple_gated
        
        data:
            dataset: old_noisy_cifar10
            noise_type: symmetric
            noise_rate: 0.5
            random_seed: 925

        model:
            architecture: resnet34
            num_classes: 10
        
        wandb:
            mode: online # "disabled" or "online"
            entity: hyounguk-shon
            project: noisy-label
            name: CIFAR10-CE-gatedNRD-randomerasing
            save_code: True
        
        trainer:
            optimizer: sgd
            init_lr: 0.1
            momentum: 0.9
            weight_decay: 1.0e-4
            lr_scheduler: multistep
            max_epoch: 200
            num_workers: 4
            batch_size: 128
            save_model: False
            loss_fn: cross_entropy
            teacher_aug: autoaugment_randomerasing
            student_aug: randomcrop
            distill_loss_fn: cross_entropy
            temperature: 5.0
            enable_amp: true
            transform_after_batching: true
            alpha: 0.5
            early_stop_epoch: 60
        """
        )
        main(config)

        # for dataset, num_classes in zip(['old_noisy_cifar10', 'old_noisy_cifar100'], [10, 100]):
        #     for noise_type in ['symmetric', 'asymmetric']:
        #         noise_rate_list = {
        #             'symmetric': [0.2, 0.5, 0.8],
        #             'asymmetric': [0.1, 0.2, 0.4]
        #         }[noise_type]
        #         for noise_rate in noise_rate_list:
        #             config['data']['dataset'] = dataset
        #             config['data']['noise_type'] = noise_type
        #             config['data']['noise_rate'] = noise_rate
        #             config['model']['num_classes'] = num_classes
        #             config['wandb']['name'] = f'ablations-CE+NRD+randomerasing'
        #             config['data']['random_seed'] = random.randint(0, 1000)
        #             executor.submit(main, deepcopy(config))