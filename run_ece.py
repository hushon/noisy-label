import yaml
import torch



if __name__ == '__main__':
    # Load YAML config
    config = yaml.safe_load(
    r"""
    data:
      dataset: noisy_cifar10
      noise_rate: 0.5
      noise_type: symmetric
      random_seed: 42
      download: true

    model:
      architecture: resnet18
      num_classes: 10

    wandb:
      mode: online # "disabled" or "online"
      entity: hyounguk-shon
      project: noisy-label
      name: NoisyCIFAR10(symm,0.5)-NROSD-CE-R34
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
      alpha: 0.5
      aug: randomcrop
      teacher_aug: autoaugment
      student_aug: randomcrop
      distill_loss_fn: kl_div
      temperature: 1.0
    """
    )
    config['wandb']['name'] = f'table1'
    from train_nrosd import main
    # from train import main

    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    pool = ProcessPoolExecutor(max_workers=max_workers)

    for dataset, num_classes in zip(['noisy_cifar10', 'noisy_cifar100'], [10, 100]):
        for noise_type in ['symmetric',]:
            noise_rate_list = {
                'symmetric': [0.2, 0.5, 0.8],
                'asymmetric': [0.1, 0.2, 0.4]
            }[noise_type]
            for noise_rate in noise_rate_list:
                for loss_fn, distill_loss_fn, init_lr in zip(['cross_entropy', ], ['kl_div', ], [0.1, ]):
                    config['data']['dataset'] = dataset
                    config['data']['noise_type'] = noise_type
                    config['data']['noise_rate'] = noise_rate
                    config['model']['num_classes'] = num_classes
                    config['trainer']['loss_fn'] = loss_fn
                    config['trainer']['distill_loss_fn'] = distill_loss_fn
                    config['trainer']['init_lr'] = init_lr
                    pool.submit(main, deepcopy(config))


    pool.shutdown()


