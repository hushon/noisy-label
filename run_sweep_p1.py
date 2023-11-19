from train import main
import yaml
import torch
import random


if __name__ == '__main__':
    config = yaml.safe_load(
    # r"""
    # method: fit_nrosd_gjs
    
    # data:
    #     dataset: old_noisy_cifar100
    #     noise_type: symmetric
    #     noise_rate: 0.2
    #     random_seed: 42

    # model:
    #     architecture: preactresnet34
    #     num_classes: 100
    
    # wandb:
    #     mode: online # "disabled" or "online"
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
    #     num_workers: 8
    #     batch_size: 128
    #     save_model: False
    #     loss_fn: gjs
    #     teacher_aug: autoaugment_randomerasing
    #     student_aug: randaugment
    #     pi: [0.3, 0.35, 0.35]
    #     enable_amp: False
    #     transform_after_batching: true
    #     early_stop_epoch: 400
    # """
    # )
    r"""
    method: fit_nrosd_gjs
    
    data:
        dataset: old_noisy_cifar100
        noise_type: asymmetric
        noise_rate: 0.4
        random_seed: 42

    model:
        architecture: preactresnet34
        num_classes: 100
    
    wandb:
        mode: online # "disabled" or "online"
        entity: hyounguk-shon
        project: noisy-label
        name: CIFAR10-CE-NRD
        save_code: True
    
    trainer:
        optimizer: sgd
        init_lr: 0.2
        momentum: 0.9
        weight_decay: 5.0e-5
        lr_scheduler: multistep_gjs
        max_epoch: 400
        num_workers: 16
        batch_size: 128
        save_model: False
        loss_fn: gjs
        teacher_aug: augmix
        student_aug: autoaugment
        pi: [0.3, 0.35, 0.35]
        enable_amp: False
        transform_after_batching: false
        early_stop_epoch: 400
    """
    )

    p1_dict = {
        'old_noisy_cifar10': {
            'symmetric': {
                0.2: [0.3, 0.35, 0.35],
                0.4: [0.9, 0.05, 0.05],
                0.6: [0.1, 0.45, 0.45],
                0.8: [0.1, 0.45, 0.45],
            },
            'asymmetric': {
                0.2: [0.3, 0.35, 0.35],
                0.4: [0.3, 0.35, 0.35],
            }
        },
        'old_noisy_cifar100': {
            'symmetric': {
                0.2: [0.3, 0.35, 0.35],
                0.4: [0.5, 0.25, 0.25],
                0.6: [0.9, 0.05, 0.05],
                0.8: [0.1, 0.45, 0.45],
            },
            'asymmetric': {
                0.2: [0.5, 0.25, 0.25],
                0.4: [0.1, 0.45, 0.45],
            }
        }
    }

    lr_dict = {
        'old_noisy_cifar10': {
            'symmetric': 0.1,
            'asymmetric': 0.1,
        },
        'old_noisy_cifar100': {
            'symmetric': 0.2,
            'asymmetric': 0.4,
        }
    }

    wd_dict = {
        'old_noisy_cifar10': {
            'symmetric': 5e-4,
            'asymmetric': 1e-3,
        },
        'old_noisy_cifar100': {
            'symmetric': 5e-5,
            'asymmetric': 1e-4,
        }
    }

    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # for pi in [[0.3, 0.35, 0.35], [0.2, 0.2, 0.6], [0.3, 0.1, 0.6]]:
        # for pi in [[0.4, 0.5, 0.1],]:
        for pi in [[0.7, 0.2, 0.1],[0.6, 0.2, 0.2],[0.5, 0.2, 0.3],[0.4, 0.2, 0.4],[0.3, 0.2, 0.5],[0.2, 0.2, 0.6],[0.1, 0.2, 0.7],]:
            config['trainer']['pi'] = pi
            config['wandb']['name'] = f'{pi=}'
            config['data']['random_seed'] = random.randint(0, 1000)
            executor.submit(main, deepcopy(config))