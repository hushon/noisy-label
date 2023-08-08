
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




def main(config):
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False))

    wandb_run = wandb.init(
        **config['wandb'],
        config=config,
    )
    wandb_run.log_code()

    train_dataset, test_dataset = get_dataset(**config["data"])

    # for multiprocessing
    if multiprocessing.current_process().name == 'MainProcess':
        torch.cuda.set_device(0)
    else:
        worker_id = multiprocessing.current_process().name
        print(f"{worker_id=}")
        worker_id = int(worker_id.split('-')[1]) - 1
        torch.cuda.set_device(worker_id)


    model = get_model(**config["model"]).cuda()

    trainer = Trainer(
                    model=model,
                    config=config['trainer'],
                    wandb_run=wandb_run,
                    )

    trainer.fit_sop(train_dataset, test_dataset)


    wandb_run.finish()



if __name__ == '__main__':
    # Load YAML config
    config = yaml.safe_load(
    r"""
    data:
      dataset: noisy_cifar10
      noise_type: symmetric
      noise_rate: 0.5
      download: true
    
    model:
      architecture: resnet34
      num_classes: 10
    
    wandb:
      mode: online # "disabled" or "online"
      entity: hyounguk-shon
      project: noisy-label
      name: table15-sop
      save_code: true
    
    trainer:
      optimizer: sgd
      init_lr: 0.02
      momentum: 0.9
      weight_decay: 1.0e-3
      lr_scheduler: multistep
      max_epoch: 200
      num_workers: 4
      batch_size: 128
      save_model: false
      loss_fn: cross_entropy
      alpha: 0.5
      aug: randomcrop
      temperature: 1.0
      enable_amp: false
      lr_u: 10
      lr_v: 10
    """
    )


    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    pool = ProcessPoolExecutor(max_workers=max_workers)

    for dataset, num_classes in zip(['noisy_cifar10', 'noisy_cifar100'], [10, 100]):
        noise_type = 'symmetric'

        for noise_rate in [0.2, 0.5, 0.8]:
            for loss_fn, init_lr in zip(['cross_entropy',], [0.02,]):
                config['data']['dataset'] = dataset
                config['data']['noise_type'] = noise_type
                config['data']['noise_rate'] = noise_rate
                config['model']['num_classes'] = num_classes
                # config['wandb']['name'] = f'{dataset}-{noise_type}-{noise_rate}-CE'
                config['trainer']['loss_fn'] = loss_fn
                config['trainer']['init_lr'] = init_lr
                if dataset == 'noisy_cifar10':
                    config['trainer']['max_epoch'] = 120
                    config['trainer']['lr_scheduler'] = 'multistep_sop_c10'
                elif dataset == 'noisy_cifar100':
                    config['trainer']['max_epoch'] = 150
                    config['trainer']['lr_scheduler'] = 'multistep_sop_c100'
                if dataset == 'noisy_cifar10':
                    config['trainer']['lr_u'] = 10
                elif dataset == 'noisy_cifar100':
                    config['trainer']['lr_u'] = 1
                if noise_type == 'symmetric':
                    config['trainer']['lr_v'] = 10
                elif noise_type == 'asymmetric':
                    config['trainer']['lr_v'] = 100
                pool.submit(main, deepcopy(config))


    pool.shutdown()


