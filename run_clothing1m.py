from train import main
import yaml
import torch



if __name__ == '__main__':
    ## Clothing1M CE ##
    # config = yaml.safe_load(
    # r""" ## Clothing1M 실험은 20시간정도 소요
    # method: fit
    
    # data:
    #     dataset: clothing1m
    #     root: /root/
    
    # model:
    #     architecture: resnet50_torchvision
    #     num_classes: 14
    #     pretrained: true

    # wandb:
    #     mode: online # "disabled" or "online"
    #     entity: hyounguk-shon
    #     project: noisy-label
    #     name: Clothing1M
    #     save_code: True
    
    # trainer:
    #     optimizer: sgd
    #     init_lr: 0.001
    #     momentum: 0.9
    #     weight_decay: 1.0e-3
    #     lr_scheduler: multistep_c1m
    #     max_epoch: 10
    #     num_workers: 4
    #     batch_size: 64
    #     save_model: false
    #     loss_fn: cross_entropy
    #     aug: randomcrop
    #     temperature: 1.0
    #     enable_amp: true
    #     transform_after_batching: false
    # """
    # )
    # main(config)


    ## Clothing1M NRD+EMA ##
    config = yaml.safe_load(
    r""" ## Clothing1M 실험은 20시간정도 소요
    method: fit_nrosd_ema
    
    data:
        dataset: clothing1m
    
    model:
        architecture: resnet50_torchvision
        num_classes: 14
        pretrained: true

    wandb:
        mode: online # "disabled" or "online"
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
        alpha: 0.5
        teacher_aug: autoaugment_randomerasing
        student_aug: randomcrop
        distill_loss_fn: cross_entropy
        temperature: 5.0
        enable_amp: true
        ema_beta: 0.25
        transform_after_batching: false
    """
    )
    main(config)