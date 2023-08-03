from train import main
import yaml
import torch



if __name__ == '__main__':
    # Load YAML config
    config = yaml.safe_load(
    r"""
    method: nrd
    
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
      save_model: true
      loss_fn: cross_entropy
      alpha: 0.5
      teacher_aug: autoaugment
      student_aug: randomcrop
      distill_loss_fn: kl_div
      temperature: 1.0
      enable_amp: false
    """
    )
    config['trainer']['save_model'] = False


    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    pool = ProcessPoolExecutor(max_workers=max_workers)

    for dataset, num_classes in zip(['noisy_cifar10', 'noisy_cifar100'], [10, 100]):
        for noise_type in ['symmetric', 'asymmetric']:
            for noise_rate in [0.2, 0.5, 0.8]:
                for loss_fn, distill_loss_fn, init_lr in zip(['cross_entropy', 'mae'], ['kl_div', 'smoothed_l1_dist'], [0.1, 0.01]):
                    config['data']['dataset'] = dataset
                    config['data']['noise_type'] = noise_type
                    config['data']['noise_rate'] = noise_rate
                    config['model']['num_classes'] = num_classes
                    config['wandb']['name'] = f'{dataset}-{noise_type}-{noise_rate}-CE'
                    config['trainer']['loss_fn'] = loss_fn
                    config['trainer']['distill_loss_fn'] = distill_loss_fn
                    config['trainer']['init_lr'] = init_lr
                    pool.submit(main, deepcopy(config))


    pool.shutdown()


