
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

    trainer.fit_nrosd_elr(train_dataset, test_dataset)


    # wandb_run.alert(
    #     title="Training finished",
    #     text="this is a test message",
    #     level=wandb.AlertLevel.INFO,
    # )
    wandb_run.finish()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Training Config', add_help=False)
#     parser.add_argument('--config', type=str, required=True, help="./configs/train_base.yaml")
#     args = parser.parse_args()


#     # Load YAML config
#     with open(args.config, 'r') as file:
#         config = yaml.safe_load(file)

#     main()


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
      architecture: resnet18
      num_classes: 10
    
    wandb:
      mode: online # "disabled" or "online"
      entity: hyounguk-shon
      project: noisy-label
      name: CIFAR10-CE-NRD-ELR
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
      save_model: false
      loss_fn: cross_entropy
      alpha: 0.5
      aug: randomcrop
      temperature: 1.0
      enable_amp: false
      loss_param:
        lam: 3.0
        beta: 0.7
    """
    )


    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    pool = ProcessPoolExecutor(max_workers=max_workers)

    for dataset, num_classes in zip(['noisy_cifar10', 'noisy_cifar100'], [10, 100]):
        for noise_type in ['symmetric', 'asymmetric']:
            noise_rate_list = {
                'symmetric': [0.2, 0.5, 0.8],
                'asymmetric': [0.1, 0.2, 0.4]
            }[noise_type]
            if dataset == 'noisy_cifar10':
                if noise_type == 'symmetric':
                    lam, beta = 3.0, 0.7
                elif noise_type == 'asymmetric':
                    lam, beta = 1.0, 0.6
            elif dataset == 'noisy_cifar100':
                lam, beta = 7.0, 0.9

            for noise_rate in noise_rate_list:
                for loss_fn, init_lr in zip(['cross_entropy', 'mae'], [0.1, 0.01]):
                    config['data']['dataset'] = dataset
                    config['data']['noise_type'] = noise_type
                    config['data']['noise_rate'] = noise_rate
                    config['model']['num_classes'] = num_classes
                    config['wandb']['name'] = f'{dataset}-{noise_type}-{noise_rate}-CE'
                    config['trainer']['loss_fn'] = loss_fn
                    config['trainer']['init_lr'] = init_lr
                    config['trainer']['loss_param']['lam'] = lam
                    config['trainer']['loss_param']['beta'] = beta

                    pool.submit(main, deepcopy(config))


    pool.shutdown()


