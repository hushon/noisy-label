
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
    # config = yaml.safe_load(
    # r"""
    # method: fit
    
    # data:
    #     dataset: old_noisy_cifar10
    #     noise_type: symmetric
    #     noise_rate: 0.5
    #     random_seed: 43
    
    # model:
    #     architecture: resnet34
    #     num_classes: 10
    
    # wandb:
    #     mode: online # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: CIFAR10-CE
    #     save_code: true
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.1
    #     momentum: 0.9
    #     weight_decay: 1.0e-4
    #     lr_scheduler: multistep
    #     max_epoch: 200
    #     num_workers: 4
    #     batch_size: 128
    #     save_model: true
    #     loss_fn: cross_entropy
    #     aug: randomcrop
    #     enable_amp: false
    #     transform_after_batching: false
    # """

    # config = yaml.safe_load(
    # r"""
    # method: fit_nrosd_v2
    
    # data:
    #     dataset: old_noisy_cifar10
    #     noise_type: symmetric
    #     noise_rate: 0.5
    #     random_seed: 42

    # model:
    #     architecture: resnet34
    #     num_classes: 10
    
    # wandb:
    #     mode: online # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: CIFAR10-CE-NRD2
    #     save_code: True
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.1
    #     momentum: 0.9
    #     weight_decay: 1.0e-4
    #     lr_scheduler: multistep
    #     max_epoch: 200
    #     num_workers: 4
    #     batch_size: 128
    #     save_model: False
    #     loss_fn: cross_entropy
    #     teacher_aug: autoaugment_randomerasing
    #     student_aug: randomcrop
    #     distill_loss_fn: cross_entropy
    #     temperature: 1.0
    #     enable_amp: False
    #     transform_after_batching: true
    #     warmup_epoch: 120
    # """
    # )
    # config = yaml.safe_load(
    # r"""
    # method: fit_nrosd
    
    # data:
    #     dataset: old_noisy_cifar10
    #     noise_type: symmetric
    #     noise_rate: 0.5
    #     random_seed: 42

    # model:
    #     architecture: preactresnet34
    #     num_classes: 10
    
    # wandb:
    #     mode: online # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: CIFAR10-CE-NRD
    #     save_code: True
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.1
    #     momentum: 0.9
    #     weight_decay: 1.0e-4
    #     lr_scheduler: multistep
    #     max_epoch: 200
    #     num_workers: 4
    #     batch_size: 128
    #     save_model: False
    #     loss_fn: cross_entropy
    #     alpha: 0.4
    #     teacher_aug: autoaugment_randomerasing
    #     student_aug: randomcrop
    #     distill_loss_fn: cross_entropy
    #     temperature: 5.0
    #     enable_amp: False
    #     transform_after_batching: true
    #     early_stop_epoch: 60
    # """
    # )
    # config = yaml.safe_load(
    # r"""
    # method: fit_nrosd_gjs
    
    # data:
    #     dataset: old_noisy_cifar10
    #     noise_type: symmetric
    #     noise_rate: 0.6
    #     random_seed: 42

    # model:
    #     architecture: preactresnet34
    #     num_classes: 10
    
    # wandb:
    #     mode: online # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: CIFAR10-CE-NRD
    #     save_code: True
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.1
    #     momentum: 0.9
    #     weight_decay: 5.0e-4
    #     lr_scheduler: multistep_gjs
    #     max_epoch: 400
    #     num_workers: 4
    #     batch_size: 128
    #     save_model: False
    #     loss_fn: cross_entropy
    #     alpha: 0.0
    #     teacher_aug: autoaugment_randomerasing
    #     student_aug: gjs
    #     distill_loss_fn: gjs
    #     temperature: 1.0
    #     enable_amp: False
    #     transform_after_batching: true
    #     early_stop_epoch: 400
    # """
    # )

    # config = yaml.safe_load(
    # r"""
    # method: fit
    
    # data:
    #     dataset: old_noisy_cifar10
    #     noise_type: symmetric
    #     noise_rate: 0.2
    #     random_seed: 42

    # model:
    #     architecture: resnet34
    #     num_classes: 10
    
    # wandb:
    #     mode: online # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: CIFAR10-GJS-NRD
    #     save_code: True
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.1
    #     momentum: 0.9
    #     weight_decay: 1.0e-4
    #     lr_scheduler: multistep
    #     max_epoch: 200
    #     num_workers: 4
    #     batch_size: 128
    #     save_model: False
    #     loss_fn: cross_entropy
    #     alpha: 0.0
    #     teacher_aug: autoaugment_randomerasing
    #     aug: randomcrop
    #     distill_loss_fn: gjs
    #     temperature: 5.0
    #     enable_amp: False
    #     transform_after_batching: true
    #     early_stop_epoch: 200
    # """
    # )
    # config = yaml.safe_load(
    # r"""
    # method: fit_nrosd_gjs
    
    # data:
    #     dataset: old_noisy_cifar100
    #     noise_type: symmetric
    #     noise_rate: 0.6
    #     random_seed: 298

    # model:
    #     architecture: preactresnet34
    #     num_classes: 100
    
    # wandb:
    #     mode: disabled # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: CIFAR10-CE-NRD
    #     save_code: True
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.2
    #     momentum: 0.9
    #     weight_decay: 5.0e-5
    #     lr_scheduler: multistep_gjs
    #     max_epoch: 400
    #     num_workers: 16
    #     batch_size: 128
    #     save_model: False
    #     loss_fn: gjs
    #     pi: [0.9, 0.05, 0.05]
    #     teacher_aug: augmix
    #     student_aug: autoaugment
    #     enable_amp: False
    #     transform_after_batching: false
    #     early_stop_epoch: 400
    # """
    # )
    config = yaml.safe_load(
    r"""
    method: fit_nrosd_gjs
    
    data:
        dataset: webvision
        root: /dev/shm/data/

    model:
        architecture: resnet50_torchvision
        num_classes: 50
        pretrained: false
    
    wandb:
        mode: online # "disabled" or "online"
        entity: hyounguk-shon
        project: noisy-label
        name: WebVision-augmix
        save_code: True
    
    trainer:
        optimizer: sgd
        init_lr: 0.4
        momentum: 0.9
        weight_decay: 1.0e-4
        lr_scheduler: steplr_gjs_webvision
        # lr_scheduler: cosine
        max_epoch: 300
        num_workers: 16
        batch_size: 128
        save_model: False
        loss_fn: gjs_jswc
        pi: [0.1, 0.4, 0.4, 0.1]
        teacher_aug: augmix
        # student_aug: autoaugment
        # teacher_aug: augmix_colorjitter
        student_aug: gjs_webvision_weak
        enable_amp: true
        transform_after_batching: false
        early_stop_epoch: 300
    """
    )
    main(config)
