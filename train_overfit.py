from train import main
import yaml
import torch
import random


if __name__ == '__main__':

    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    config = yaml.safe_load(
    r"""
    method: fit_gjs
    
    data:
        dataset: old_noisy_cifar100
        noise_type: symmetric
        noise_rate: 0.8
        random_seed: 42

    model:
        architecture: preactresnet34
        # architecture: resnet34
        num_classes: 100
    
    wandb:
        mode: online # "disabled" or "online"
        entity: hyounguk-shon
        project: noisy-label
        name: CIFAR100-GJS-overfit
        save_code: True
    
    trainer:
        optimizer: sgd
        init_lr: 0.2
        momentum: 0.9
        # optimizer: adam
        # init_lr: 0.0001
        weight_decay: 5.0e-5
        # lr_scheduler: multistep_gjs
        lr_scheduler: multistep_gjs_overfit
        max_epoch: 1000
        num_workers: 16
        batch_size: 128
        save_model: False
        loss_fn: gjs
        teacher_aug: gjs
        student_aug: gjs
        pi: [0.1, 0.45, 0.45]
        enable_amp: false
        transform_after_batching: false
        early_stop_epoch: 1000
    """
    )
    main(config)