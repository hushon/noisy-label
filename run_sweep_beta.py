from train import main
import yaml
import torch
import random


if __name__ == '__main__':
    # Load YAML config
    config = yaml.safe_load(
    r"""
    method: fit_nrosd_ema
    
    data:
        dataset: old_noisy_cifar10
        noise_type: symmetric
        noise_rate: 0.5
        random_seed: 42
    
    model:
        architecture: resnet34
        num_classes: 10
    
    wandb:
        mode: online # "disabled" or "online"
        entity: hyounguk-shon
        project: noisy-label
        name: CIFAR10-CE-NRD-EMA
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
        alpha: 0.2
        teacher_aug: autoaugment
        student_aug: randomcrop
        distill_loss_fn: cross_entropy
        temperature: 1.0
        enable_amp: False
        ema_beta: 0.9999
        transform_after_batching: false
    """
    )

    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ema_beta in [0.999, 0.9995, 0.9999, 0.99995, 0.99999]:
            config['trainer']['ema_beta'] = ema_beta
            config['wandb']['name'] = f'{ema_beta=}'
            config['data']['random_seed'] = random.randint(0, 1000)
            executor.submit(main, deepcopy(config))
