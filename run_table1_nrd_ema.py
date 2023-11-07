from train import main
import yaml
import torch



if __name__ == '__main__':
    # Load YAML config
    config = yaml.safe_load(
    r"""
    method: fit_nrosd_ema
    
    data:
        dataset: old_noisy_cifar100
        noise_type: symmetric
        noise_rate: 0.5
    
    model:
        architecture: resnet34
        num_classes: 100
    
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
        ema_beta: 0.9995
        transform_after_batching: false
    """
    # r"""
    # method: fit_nrosd_ema_multiple
    
    # data:
    #     dataset: old_noisy_cifar100
    #     noise_type: symmetric
    #     noise_rate: 0.5
    
    # model:
    #     architecture: resnet34
    #     num_classes: 100
    
    # wandb:
    #     mode: online # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: CE-NRD-EMA
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
    #     ema_beta: 0.9995
    #     n_views: 10
        # transform_after_batching: false
    # """
    )

    # main(config)
    # exit()

    # run a list of configs in parallel
    from concurrent.futures import ProcessPoolExecutor
    from copy import deepcopy

    max_workers = torch.cuda.device_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # for ema_beta in [0.999, 0.9995, 0.9999, 0.99995, 0.99999]:
        #     config['trainer']['ema_beta'] = ema_beta
        #     config['wandb']['name'] = f'{ema_beta=}'
        #     executor.submit(main, deepcopy(config))
        # for alpha in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
        #     config['trainer']['alpha'] = alpha
        #     config['wandb']['name'] = f'{alpha=}'
        #     executor.submit(main, deepcopy(config))


        for dataset, num_classes in zip(['old_noisy_cifar10', 'old_noisy_cifar100'], [10, 100]):
            for noise_type in ['symmetric', 'asymmetric']:
                noise_rate_list = {
                    'symmetric': [0.2, 0.5, 0.8],
                    'asymmetric': [0.1, 0.2, 0.4]
                }[noise_type]
                for noise_rate in noise_rate_list:
                    for loss_fn, distill_loss_fn, init_lr in zip(['cross_entropy', 'mae'], ['cross_entropy', 'smoothed_l1_dist'], [0.1, 0.01]):
                        if loss_fn == 'cross_entropy':
                            pass
                        else:
                            continue
                        config['data']['dataset'] = dataset
                        config['data']['noise_type'] = noise_type
                        config['data']['noise_rate'] = noise_rate
                        config['model']['num_classes'] = num_classes
                        config['wandb']['name'] = f'{dataset}-{noise_type}-{noise_rate}-CE-NRD-EMA'
                        config['trainer']['loss_fn'] = loss_fn
                        config['trainer']['distill_loss_fn'] = distill_loss_fn
                        config['trainer']['init_lr'] = init_lr
                        executor.submit(main, deepcopy(config))