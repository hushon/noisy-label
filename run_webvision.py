from train import main
import yaml
import torch



if __name__ == '__main__':
    # ## WebVision CE
    # config = yaml.safe_load(
    # r""" ## WebVision 실험은 3.5시간정도 소요
    # method: fit
    
    # data:
    #     dataset: webvision
    #     root: /root/
    
    # model:
    #     architecture: resnet50_torchvision
    #     num_classes: 50
    #     pretrained: false
    
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
    #     aug: randomcrop
    #     distill_loss_fn: cross_entropy
    #     enable_amp: true
    #     transform_after_batching: false
    # """
    # )
    # main(config)


    ## WebVision NRD+EMA
    config = yaml.safe_load(
    r""" ## WebVision 실험은 3.5시간정도 소요
    method: fit_nrosd_ema
    
    data:
        dataset: webvision
        root: /root/
    
    model:
        architecture: resnet50_torchvision
        num_classes: 50
        pretrained: false
    
    wandb:
        mode: online # "disabled" or "online"
        entity: hyounguk-shon
        project: noisy-label
        name: WebVision
        save_code: True
    
    trainer:
        optimizer: sgd
        init_lr: 0.1
        momentum: 0.9
        weight_decay: 1.0e-4
        lr_scheduler: cosine
        max_epoch: 120
        num_workers: 4
        batch_size: 128
        save_model: false
        loss_fn: cross_entropy
        alpha: 0.2
        teacher_aug: autoaugment
        student_aug: randomcrop
        distill_loss_fn: cross_entropy
        temperature: 1.0
        enable_amp: true
        ema_beta: 0.9995
        transform_after_batching: false
    """
    )
    main(config)