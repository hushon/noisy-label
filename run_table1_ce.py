from train import main
import yaml
import torch
import random


if __name__ == '__main__':
    # Load YAML config
    config = yaml.safe_load(
    r"""
    method: fit
    
    data:
      dataset: noisy_cifar10
      noise_type: symmetric
      noise_rate: 0.5
      random_seed: 43
    
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
      init_lr: 0.1
      momentum: 0.9
      weight_decay: 1.0e-4
      lr_scheduler: multistep
      max_epoch: 200
      num_workers: 4
      batch_size: 128
      save_model: false
      loss_fn: cross_entropy
      aug: randomcrop
      enable_amp: false
      transform_after_batching: false
    """
    )


    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
        #             config['wandb']['name'] = f'{dataset}-{noise_type}-{noise_rate}-CE'
        #             config['data']['random_seed'] = random.randint(0, 1000)
        #             executor.submit(main, deepcopy(config))
        for init_lr in [0.01, 0.05, 0.1, 0.5, 1.0]:
            config['trainer']['init_lr'] = init_lr
            config['wandb']['name'] = f'CIFAR10-CE-{init_lr}'
            config['data']['random_seed'] = random.randint(0, 1000)
            executor.submit(main, deepcopy(config))

