from train import main
import yaml
import torch



if __name__ == '__main__':
    # ## WebVision NRD+EMA
    # config = yaml.safe_load(
    # r""" ## WebVision 실험은 3.5시간정도 소요
    # method: fit_nrosd_ema
    
    # data:
    #     dataset: webvision
    
    # model:
    #     architecture: resnet50_torchvision
    #     num_classes: 50
    #     pretrained: false # 이거 왜 false?
    
    # wandb:
    #     mode: online # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: WebVision
    #     save_code: True
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.1
    #     momentum: 0.9
    #     weight_decay: 1.0e-4
    #     lr_scheduler: cosine
    #     max_epoch: 120
    #     num_workers: 4
    #     batch_size: 128
    #     save_model: false
    #     loss_fn: cross_entropy
    #     alpha: 0.2
    #     teacher_aug: autoaugment
    #     student_aug: randomcrop
    #     distill_loss_fn: cross_entropy
    #     temperature: 1.0
    #     enable_amp: true
    #     ema_beta: 0.9995
    #     transform_after_batching: false
    # """
    # )
    # main(config)



    ## Clothing1M NRD+EMA ##
    config = yaml.safe_load(
    r""" ## Clothing1M 실험은 20시간정도 소요
    method: fit
    
    data:
        dataset: clothing1m
    
    model:
        architecture: resnet50_torchvision
        num_classes: 14
        pretrained: true

    wandb:
        mode: disabled # "disabled" or "online"
        entity: hyounguk-shon
        project: noisy-label
        name: Clothing1M
        save_code: True
    
    trainer:
        optimizer: sgd
        init_lr: 0.001
        momentum: 0.9
        weight_decay: 1.0e-4
        lr_scheduler: multistep_c1m
        max_epoch: 10
        num_workers: 4
        batch_size: 64
        save_model: false
        loss_fn: cross_entropy
        aug: randomcrop
        temperature: 1.0
        enable_amp: true
        transform_after_batching: false
    """
    )
    main(config)

    # config = yaml.safe_load(
    # r""" ## Clothing1M 실험은 20시간정도 소요
    # method: fit_nrosd_ema
    
    # data:
    #     dataset: clothing1m
    
    # model:
    #     architecture: resnet50_torchvision
    #     num_classes: 14
    #     pretrained: true

    # wandb:
    #     mode: disabled # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: Clothing1M
    #     save_code: True
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.001
    #     momentum: 0.9
    #     weight_decay: 1.0e-4
    #     lr_scheduler: multistep_c1m
    #     max_epoch: 10
    #     num_workers: 4
    #     batch_size: 64
    #     save_model: false
    #     loss_fn: cross_entropy
    #     alpha: 0.2
    #     teacher_aug: autoaugment
    #     student_aug: randomcrop
    #     distill_loss_fn: cross_entropy
    #     temperature: 1.0
    #     enable_amp: true
    #     ema_beta: 0.9995
    #     transform_after_batching: false
    # """
    # )
    # main(config)