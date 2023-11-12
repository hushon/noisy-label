from train import main
import yaml
import torch
import random


if __name__ == '__main__':
    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # # ==== CE ====
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
        #     save_model: false
        #     loss_fn: cross_entropy
        #     aug: randomcrop
        #     enable_amp: false
        #     transform_after_batching: false
        # """
        # )
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
        #             config['wandb']['name'] = f'ablations-CE'
        #             config['data']['random_seed'] = random.randint(0, 1000)
        #             executor.submit(main, deepcopy(config))

        # # ==== CE+NRD ====
        # config = yaml.safe_load(
        # r"""
        # method: fit_nrosd
        
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
        #     alpha: 0.5
        #     teacher_aug: autoaugment
        #     student_aug: randomcrop
        #     distill_loss_fn: cross_entropy
        #     temperature: 1.0
        #     enable_amp: False
        #     transform_after_batching: false
        # """
        # )
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
        #             config['wandb']['name'] = f'ablations-CE+NRD'
        #             config['data']['random_seed'] = random.randint(0, 1000)
        #             executor.submit(main, deepcopy(config))

        # # ==== CE+NRD + randomerasing ====
        # config = yaml.safe_load(
        # r"""
        # method: fit_nrosd
        
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
        #     alpha: 0.5
        #     teacher_aug: autoaugment_randomerasing
        #     student_aug: randomcrop
        #     distill_loss_fn: cross_entropy
        #     temperature: 1.0
        #     enable_amp: False
        #     transform_after_batching: false
        # """
        # )
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


        # # ==== CE+NRD+EMA ====
        # config = yaml.safe_load(
        # r"""
        # method: fit_nrosd_ema
        
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
        #     name: CIFAR10-CE-NRD-EMA
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
        #     alpha: 0.2
        #     teacher_aug: autoaugment
        #     student_aug: randomcrop
        #     distill_loss_fn: cross_entropy
        #     temperature: 1.0
        #     enable_amp: False
        #     ema_beta: 0.25
        #     transform_after_batching: false
        # """
        # )
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
        #             config['wandb']['name'] = f'ablations-CE+NRD+EMA'
        #             config['data']['random_seed'] = random.randint(0, 1000)
        #             executor.submit(main, deepcopy(config))

        # # ==== CE+NRD+EMA + randomerasing ====
        # config = yaml.safe_load(
        # r"""
        # method: fit_nrosd_ema
        
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
        #     name: CIFAR10-CE-NRD-EMA
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
        #     alpha: 0.2
        #     teacher_aug: autoaugment_randomerasing
        #     student_aug: randomcrop
        #     distill_loss_fn: cross_entropy
        #     temperature: 1.0
        #     enable_amp: False
        #     ema_beta: 0.25
        #     transform_after_batching: false
        # """
        # )
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
        #             config['wandb']['name'] = f'ablations-CE+NRD+EMA'
        #             config['data']['random_seed'] = random.randint(0, 1000)
        #             executor.submit(main, deepcopy(config))




        # # ==== CE+NRD+dropout ====
        # config = yaml.safe_load(
        # r"""
        # method: fit_nrosd_dropout
        
        # data:
        #     dataset: old_noisy_cifar10
        #     noise_type: symmetric
        #     noise_rate: 0.5
        #     random_seed: 42
        
        # model:
        #     architecture: resnet34_dropout
        #     num_classes: 10
        
        # wandb:
        #     mode: online # "disabled" or "online"
        #     entity: hyounguk-shon
        #     project: noisy-label
        #     name: CIFAR10-CE-NRD-EMA
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
        #     alpha: 0.5
        #     teacher_aug: autoaugment
        #     student_aug: randomcrop
        #     distill_loss_fn: cross_entropy
        #     temperature: 1.0
        #     enable_amp: False
        #     transform_after_batching: false
        # """
        # )
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
        #             config['wandb']['name'] = f'ablations-CE+NRD+dropout'
        #             config['data']['random_seed'] = random.randint(0, 1000)
        #             executor.submit(main, deepcopy(config))


        # # ==== CE+NRD+EMA+dropout ====
        # config = yaml.safe_load(
        # r"""
        # method: fit_nrosd_ema_dropout
        
        # data:
        #     dataset: old_noisy_cifar10
        #     noise_type: symmetric
        #     noise_rate: 0.5
        #     random_seed: 43
        
        # model:
        #     architecture: resnet34_dropout
        #     num_classes: 10
        
        # wandb:
        #     mode: online # "disabled" or "online"
        #     entity: hyounguk-shon
        #     project: noisy-label
        #     name: CIFAR10-CE-NRD-EMA
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
        #     alpha: 0.2
        #     teacher_aug: autoaugment
        #     student_aug: randomcrop
        #     distill_loss_fn: cross_entropy
        #     temperature: 1.0
        #     enable_amp: False
        #     ema_beta: 0.9999
        #     transform_after_batching: false
        # """
        # )
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
        #             config['wandb']['name'] = f'ablations-CE+NRD+EMA+dropout'
        #             config['data']['random_seed'] = random.randint(0, 1000)
        #             executor.submit(main, deepcopy(config))
