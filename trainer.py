import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm.auto as tqdm
import os
import wandb
from wandb.sdk.wandb_run import Run
import pdb
from models import MeanAbsoluteError, ReverseCrossEntropyLoss, SymmetricCrossEntropyLoss, \
    GeneralizedCrossEntropyLoss, CrossEntropyDistillationLoss, JensenShannonDivergenceWeightedScaled, JensenShannonDivergenceWeightedCustom
from models import KLDivDistillationLoss, L1DistillationLoss, SmoothL1DistillationLoss, Normalize2D
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2
import numpy as np
import datasets
from collections import defaultdict
import einops


class Trainer:
    def __init__(self, model: nn.Module, config: dict=None, wandb_run: Run=None, device='cuda:0'):
        self.device = torch.device(device)
        self.model = model
        # self.model = torch.compile(self.model)
        # self.model = torch.jit.script(self.model)
        self.model = self.model.to(self.device)
        self.config = config
        self.wandb_run = wandb_run

    def get_optimizer(
            self,
            model: nn.Module
        ) -> torch.optim.Optimizer:
        match self.config["optimizer"]:
            case "sgd":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.config["init_lr"],
                    momentum=self.config["momentum"],
                    weight_decay=self.config["weight_decay"],
                    nesterov=True,
                )
            case "adam":
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.config["init_lr"],
                    weight_decay=self.config["weight_decay"],
                )
            case _:
                raise NotImplementedError(self.config["optimizer"])
        return optimizer

    def get_lr_scheduler(
            self,
            optimizer: torch.optim.Optimizer
        ) -> torch.optim.lr_scheduler.LRScheduler:
        match self.config["lr_scheduler"]:
            case "cosine":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config["max_epoch"],
                    eta_min=0.0,
                )
            case "multistep":
                n = self.config["max_epoch"] // 10
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[3*n, 6*n, 8*n],
                    gamma=0.2,
                )
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                #     optimizer,
                #     milestones=[150],
                #     gamma=0.1,
                # )
                # breakpoint()
            case "multistep2":
                n = self.config["max_epoch"] // 10
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[4*n, 6*n],
                    gamma=0.1,
                )
            case "multistep_c1m":
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[self.config["max_epoch"] // 2,],
                    gamma=0.1,
                )
            case "multistep_gjs":
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[200, 300],
                    gamma=0.1,
                )
            case "steplr_gjs_webvision":
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=1,
                    gamma=0.97,
                )
            case _:
                raise NotImplementedError(self.config["lr_scheduler"])
        return lr_scheduler

    def get_loss_fn(self, fn_name) -> nn.Module:
        match fn_name:
            case "cross_entropy":
                fn = nn.CrossEntropyLoss(reduction="none")
            case "mae":
                fn = MeanAbsoluteError(reduction="none")
            case "reverse_cross_entropy":
                fn = ReverseCrossEntropyLoss(
                                    num_classes=self.model.fc.out_features,
                                    reduction="none",
                                    )
            case "symmetric_cross_entropy":
                fn = SymmetricCrossEntropyLoss(
                                    alpha=self.config["loss_param"]["alpha"],
                                    beta=self.config["loss_param"]["beta"],
                                    num_classes=self.model.fc.out_features,
                                    reduction="none",
                                    )
            case "generalized_cross_entropy":
                fn = GeneralizedCrossEntropyLoss(
                                    num_classes=self.model.fc.out_features,
                                    q=self.config["loss_param"]["q"],
                                    reduction="none",
                                    )
            case "gjs":
                fn = JensenShannonDivergenceWeightedScaled(self.config['pi'])
            case "gjs_jswc":
                fn = JensenShannonDivergenceWeightedCustom(self.config['pi'])
            case _:
                raise NotImplementedError(fn_name)
        return fn

    @staticmethod
    def get_distill_loss_fn(fn_name, temperature=1.0) -> nn.Module:
        match fn_name:
            case "kl_div":
                fn = KLDivDistillationLoss(temperature, reduction="none")
            case "l1_dist":
                fn = L1DistillationLoss(temperature, reduction="none")
            case "smoothed_l1_dist":
                fn = SmoothL1DistillationLoss(temperature, reduction="none")
            case "cross_entropy":
                fn = CrossEntropyDistillationLoss(temperature, reduction="none")
            case _:
                raise NotImplementedError(fn_name)
        return fn

    def get_dataloader(self, dataset: Dataset, train=True) -> DataLoader:
        # return DataLoader(
        return datasets.MultiEpochsDataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=train,
            num_workers=self.config["num_workers"],
            drop_last=train,
        )

    def fit(self, train_dataset: Dataset, val_dataset: Dataset):
        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            transform = datasets.get_transform(self.config['aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(f'checkpoints',
                                  type='model',
                                  metadata=self.wandb_run.config['model'])

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data = batch["image"].to(self.device)
                if self.config['transform_after_batching']:
                    data = transform(data)
                data = normalize(data)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    output = self.model(data)
                    loss = self.criterion(output, target).mean()
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                )

            lr_scheduler.step()
            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def fit_nrosd(self, train_dataset: Dataset, val_dataset: Dataset):
        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            if epoch == self.config['early_stop_epoch']: # use early stopping
                break
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), transform2(data2)
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    # self.model.train()
                    output = self.model(data)
                    with torch.no_grad():
                        # self.model.eval()
                        output_teacher = self.model(data2)
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def fit_nrosd_gjs(self, train_dataset: Dataset, val_dataset: Dataset):
        if self.config['transform_after_batching']:
            raise ValueError
        else:
            train_transforms = [
                datasets.get_transform(self.config['student_aug'], train_dataset),
                datasets.get_transform(self.config['student_aug'], train_dataset),
                datasets.get_transform(self.config['teacher_aug'], train_dataset),
            ]

        from datasets import MultiTransformDataset
        train_dataset = MultiTransformDataset(train_dataset, train_transforms)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn('cross_entropy').to(self.device)
        distill_loss_fn = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        normalize = datasets.get_normalization(train_dataset.dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            if epoch == self.config['early_stop_epoch']: # use early stopping
                break
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target, target_gt = batch["target"].to(self.device), batch["target_gt"].to(self.device)
                batch["image"] = [normalize(x.to(self.device)) for x in batch["image"]]
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    # self.model.train()
                    outputs = [self.model(x) for x in batch["image"]]
                    outputs[-1] = outputs[-1].detach()
                    target_loss = torch.tensor(0.0)
                    distill_loss = distill_loss_fn([*outputs], target).mean()
                    loss = distill_loss
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(outputs[0], target),
                    t5acc=calculate_accuracy(outputs[0], target, k=5),
                    teacher_gt_acc=calculate_accuracy(outputs[-1], target_gt),
                    student_gt_acc=calculate_accuracy(outputs[0], target_gt),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def fit_nrosd_ema(self, train_dataset: Dataset, val_dataset: Dataset):
        from ema_pytorch import EMA
        ema_model = EMA(
            self.model,
            beta=self.config['ema_beta'],
            update_after_step=100,
            update_every=10,
        ).train()
        # ema_model = EMA(self.model).eval()

        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), transform2(data2)
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    output = self.model(data)
                    with torch.no_grad():
                        output_teacher = ema_model(data2)
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                ema_model.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                # artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)


    def fit_nrosd_ema_instance(self, train_dataset: Dataset, val_dataset: Dataset):

        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        tabular_ema = TabularEMA(
            num_samples=len(train_dataset),
            num_classes=self.model.fc.out_features,
            beta=self.config['ema_beta'],
        ).to(self.device)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target, target_gt = batch["target"].to(self.device), batch["target_gt"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), transform2(data2)
                data, data2 = normalize(data), normalize(data2)
                indices = batch['index'].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    output = self.model(data)
                    with torch.no_grad():
                        output_teacher = self.model(data2)
                        # output_teacher = tabular_ema(output_teacher, indices)
                        output_teacher = tabular_ema(output_teacher.softmax(-1), indices).log()
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                # artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)


    def fit_nrosd_multiple(self, train_dataset: Dataset, val_dataset: Dataset):
        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), torch.stack([transform2(data2) for _ in range(10)], dim=0)
                else:
                    raise NotImplementedError
                # b = data.shape[0]
                # data2 = data2.view(-1, *data2.shape[2:])
                data2 = einops.rearrange(data2, 'n b c h w -> (n b) c h w')
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    # self.model.train()
                    output = self.model(data)
                    with torch.no_grad():
                        # self.model.eval()
                        output_teacher = self.model(data2)
                        # output_teacher = output_teacher.view(10, b, *output_teacher.shape[1:]).mean(0)
                        output_teacher = einops.rearrange(output_teacher, '(n b) c -> n b c', n=10).mean(0)
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def fit_nrosd_ema_multiple(self, train_dataset: Dataset, val_dataset: Dataset):
        from ema_pytorch import EMA
        ema_model = EMA(
            self.model,
            beta=self.config['ema_beta'],
            update_after_step=100,
            update_every=10,
        ).train()
        # ema_model = EMA(self.model).eval()

        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']
        n_views = self.config['n_views']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), torch.stack([transform2(data2) for _ in range(n_views)], dim=0)
                else:
                    raise NotImplementedError
                # b = data.shape[0]
                # data2 = data2.view(-1, *data2.shape[2:])
                data2 = einops.rearrange(data2, 'n b c h w -> (n b) c h w')
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    output = self.model(data)
                    with torch.no_grad():
                        output_teacher = ema_model(data2)
                        # output_teacher = output_teacher.view(10, b, *output_teacher.shape[1:]).mean(0)
                        output_teacher = einops.rearrange(output_teacher, '(n b) c -> n b c', n=n_views).mean(0)
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                ema_model.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                # artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)


    def fit_nrosd_ema_instance_multiple(self, train_dataset: Dataset, val_dataset: Dataset):

        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        tabular_ema = TabularEMA(
            num_samples=len(train_dataset),
            num_classes=self.model.fc.out_features,
            beta=self.config['ema_beta'],
        ).to(self.device)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target, target_gt = batch["target"].to(self.device), batch["target_gt"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), torch.stack([transform2(data2) for _ in range(10)], dim=0)
                else:
                    raise NotImplementedError
                # b = data.shape[0]
                # data2 = data2.view(-1, *data2.shape[2:])
                data2 = einops.rearrange(data2, 'n b c h w -> (n b) c h w')
                data, data2 = transform1(data), transform2(data2)
                data, data2 = normalize(data), normalize(data2)
                indices = batch['index'].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    output = self.model(data)
                    with torch.no_grad():
                        output_teacher = self.model(data2)
                        # output_teacher = output_teacher.view(10, b, *output_teacher.shape[1:]).mean(0)
                        output_teacher = einops.rearrange(output_teacher, '(n b) c -> n b c', n=10).mean(0)
                        # output_teacher = tabular_ema(output_teacher, indices)
                        output_teacher = tabular_ema(output_teacher.softmax(-1), indices).log()
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                # artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)


    def fit_nrosd_dropout(self, train_dataset: Dataset, val_dataset: Dataset):
        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), transform2(data2)
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    output = self.model(data)
                    with torch.no_grad():
                        output_teacher = self.model(data2)
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                # artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)


    def fit_nrosd_ema_dropout(self, train_dataset: Dataset, val_dataset: Dataset):
        from ema_pytorch import EMA
        ema_model = EMA(
            self.model,
            beta=self.config['ema_beta'],
            update_after_step=100,
            update_every=10,
        ).train()
        # ema_model = EMA(self.model).eval()

        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), transform2(data2)
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    output = self.model(data)
                    with torch.no_grad():
                        output_teacher = ema_model(data2)
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                ema_model.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                # artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)


    def fit_nrosd_v2(self, train_dataset: Dataset, val_dataset: Dataset):
        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), torch.stack([transform2(data2) for _ in range(10)], dim=0)
                else:
                    raise NotImplementedError
                data2 = einops.rearrange(data2, 'n b c h w -> (n b) c h w')
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    # self.model.train()
                    output = self.model(data)
                    # loss = target_loss * alpha + distill_loss * (1.0-alpha)
                    if epoch < self.config['warmup_epoch']:
                        target_loss = self.criterion(output, target).mean()
                        distill_loss = torch.tensor(0.0)
                        loss = target_loss
                    else:
                        with torch.no_grad():
                            # self.model.eval()
                            output_teacher = self.model(data2)
                            output_teacher = einops.rearrange(output_teacher, '(n b) c -> n b c', n=10).mean(0)
                        target_loss = torch.tensor(0.0)
                        distill_loss = distill_criterion(output, output_teacher).mean()
                        loss = distill_loss
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)


    def fit_nrosd_offline(self, train_dataset: Dataset, val_dataset: Dataset):
        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)

        normalize = datasets.get_normalization(train_dataset).to(self.device)
        import copy
        teacher_model = copy.deepcopy(self.model).to(self.device).eval()


        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), torch.stack([transform2(data2) for _ in range(10)], dim=0)
                else:
                    raise NotImplementedError
                data2 = einops.rearrange(data2, 'n b c h w -> (n b) c h w')
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    # self.model.train()
                    output = self.model(data)
                    # loss = target_loss * alpha + distill_loss * (1.0-alpha)
                    with torch.no_grad():
                        # self.model.eval()
                        output_teacher = teacher_model(data2)
                        output_teacher = einops.rearrange(output_teacher, '(n b) c -> n b c', n=10).mean(0)
                    target_loss = torch.tensor(0.0)
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = distill_loss
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach()
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def fit_nrosd_multiple_gated(self, train_dataset: Dataset, val_dataset: Dataset):
        if self.config['transform_after_batching']:
            train_dataset.transform = datasets.get_transform('none', train_dataset)
            train_dataset.transform2 = datasets.get_transform('none', train_dataset)
            transform1 = datasets.get_transform(self.config['student_aug'], train_dataset)
            transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)
        else:
            train_dataset.transform = datasets.get_transform(self.config['student_aug'], train_dataset)
            train_dataset.transform2 = datasets.get_transform(self.config['teacher_aug'], train_dataset)

        val_dataset.transform = datasets.get_transform('none', val_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)
        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(
            'checkpoints',
            type='model',
            metadata=self.wandb_run.config['model']
        )

        distill_criterion = self.get_distill_loss_fn(
            self.config["distill_loss_fn"],
            self.config['temperature']
        ).to(self.device)
        alpha = self.config['alpha']

        normalize = datasets.get_normalization(train_dataset).to(self.device)

        val_t1acc_best = 0.0

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = AverageMeter()
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                if self.config['transform_after_batching']:
                    data, data2 = transform1(data), torch.stack([transform2(data2) for _ in range(10)], dim=0)
                else:
                    raise NotImplementedError
                data2 = einops.rearrange(data2, 'n b c h w -> (n b) c h w')
                data, data2 = normalize(data), normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    # self.model.train()
                    output = self.model(data)
                    with torch.no_grad():
                        # self.model.eval()
                        output_teacher = self.model(data2)
                        output_teacher = einops.rearrange(output_teacher, '(n b) c -> n b c', n=10).mean(0)
                    target_loss = self.criterion(output, target)
                    distill_loss = distill_criterion(output, output_teacher)
                    gating = (torch.nn.functional.cross_entropy(output_teacher, output_teacher.softmax(-1), reduction='none').sigmoid().detach() - 0.5)*2
                    loss = (target_loss * gating).mean() * alpha + (distill_loss * (1.0-gating)).mean() * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats.update(
                    target.size(0),
                    loss=loss.detach(),
                    t1acc=calculate_accuracy(output, target),
                    t5acc=calculate_accuracy(output, target, k=5),
                    target_loss=target_loss.detach(),
                    distill_loss=distill_loss.detach(),
                    gating=gating,
                )
            lr_scheduler.step()

            train_stats = train_stats.get_average()
            val_stats = self._evaluate(val_dataloader)
            if val_stats['t1acc'] > val_t1acc_best: # update best t1acc
                val_t1acc_best = val_stats['t1acc']
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                    'val_t1acc_best': val_t1acc_best,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                print(f"SAVED MODEL")
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)


    def distill(self, train_dataset: Dataset, val_dataset: Dataset, teacher_model: nn.Module):
        train_dataset.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(np.array(x)).permute(2,0,1)),
        ]) # output is a (3, 32, 32) uint8 tensor
        transform_teacher = transforms.Compose([
            transforms_v2.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])
        transform_student = transforms.Compose([
            transforms_v2.RandomCrop(32, padding=4),
            transforms_v2.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)

        teacher_model = torch.jit.script(teacher_model)
        # teacher_model = torch.compile(teacher_model)

        for epoch in tqdm.trange(self.config["max_epoch"]):
            train_stats = {
                "loss": [],
                "t1acc": [],
                "t5acc": [],
            }
            self.model.train()
            teacher_model.eval() # 이거 train 으로 두면 성능 더 오를듯?
            for batch in train_dataloader:
                data, target = batch["image"].to(self.device), batch["target"].to(self.device)
                with torch.no_grad():
                    output_teacher = teacher_model(transform_teacher(data))
                output = self.model(transform_student(data))
                loss = (self.config['temperature'] ** 2) * F.kl_div(
                    output.div(self.config['temperature']).log_softmax(-1),
                    output_teacher.div_(self.config['temperature']).softmax(-1),
                    reduction='batchmean'
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_stats["loss"].append(loss.detach())
                train_stats["t1acc"].append(calculate_accuracy(output, target))
                train_stats["t5acc"].append(calculate_accuracy(output, target, k=5))
            lr_scheduler.step()
            train_stats["loss"] = torch.stack(train_stats["loss"]).mean().item()
            train_stats["t1acc"] = torch.stack(train_stats["t1acc"]).mean().item()
            train_stats["t5acc"] = torch.stack(train_stats["t5acc"]).mean().item()

            val_stats = self._evaluate(val_dataloader)
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                self.wandb_run.save(filepath)

    def fit_nrosd_hardlabel(self, train_dataset: Dataset, val_dataset: Dataset):
        train_dataset.transform = self.get_transform(self.config['student_aug'], train_dataset)
        train_dataset.transform2 = self.get_transform(self.config['teacher_aug'], train_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(f'checkpoints',
                                  type='model',
                                  metadata=self.wandb_run.config['model'])

        distill_criterion = self.get_distill_loss_fn(
                                            self.config["distill_loss_fn"],
                                            self.config['temperature']
                                            ).to(self.device)
        alpha = self.config['alpha']

        normalize = self.get_normalization(train_dataset).to(self.device)

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = {
                "loss": [],
                "t1acc": [],
                "t5acc": [],
                "target_loss": [],
                "distill_loss": [],
            }
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"].to(self.device), batch['image2'].to(self.device)
                data = normalize(data)
                data2 = normalize(data2)
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    self.model.train()
                    output = self.model(data)
                    with torch.no_grad():
                        self.model.eval()
                        output_teacher = self.model(data2)
                    target_loss = self.criterion(output, target).mean()
                    output_teacher = output_teacher.argmax(-1)
                    output_teacher = F.one_hot(output_teacher, num_classes=self.model.fc.out_features).float()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats["loss"].append(loss.detach())
                train_stats["t1acc"].append(calculate_accuracy(output, target))
                train_stats["t5acc"].append(calculate_accuracy(output, target, k=5))
                train_stats["target_loss"].append(target_loss.detach())
                train_stats["distill_loss"].append(distill_loss.detach())
            lr_scheduler.step()
            train_stats["loss"] = torch.stack(train_stats["loss"]).mean().item()
            train_stats["t1acc"] = torch.stack(train_stats["t1acc"]).mean().item()
            train_stats["t5acc"] = torch.stack(train_stats["t5acc"]).mean().item()
            train_stats["target_loss"] = torch.stack(train_stats["target_loss"]).mean().item()
            train_stats["distill_loss"] = torch.stack(train_stats["distill_loss"]).mean().item()

            val_stats = self._evaluate(val_dataloader)
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def distill_perturb(self, train_dataset: Dataset, val_dataset: Dataset, teacher_model: nn.Module, N_aug: int = 100):

        train_dataset.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(np.array(x)).permute(2,0,1)),
        ]) # output is a (3, 32, 32) uint8 tensor
        # TODO: change x.repeat(N, 1, 1, 1) => torch.stack([x for _ in range(N)], dim=0).reshape((-1, *x.shape[1:]))
        transform_teacher = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(N_aug, 1, 1, 1)), # It makes data[i*batch:(i+1)batch] == data[(i+1)*batch:(i+2)*batch]
            transforms_v2.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])
        transform_student = transforms.Compose([
            transforms_v2.RandomCrop(32, padding=4),
            transforms_v2.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)

        teacher_model = torch.jit.script(teacher_model)
        # teacher_model = torch.compile(teacher_model)
        for epoch in tqdm.trange(self.config["max_epoch"]):
            train_stats = {
                "loss": [],
                "t1acc": [],
                "t5acc": [],
            }
            self.model.train()
            teacher_model.eval() # 이거 train 으로 두면 성능 더 오를듯?
            for batch in train_dataloader:
                data, target = batch["image"].to(self.device), batch["target"].to(self.device)
                with torch.no_grad():
                    output_teacher = teacher_model(transform_teacher(data)) # data shape: (N, 3, 32, 32)
                    output_teacher = output_teacher.reshape((N_aug, -1, output_teacher.shape[-1])).mean(dim=0) # dim[1] == batch_size
                output = self.model(transform_student(data))
                loss = (self.config['temperature'] ** 2) * F.kl_div(
                    output.div(self.config['temperature']).log_softmax(-1),
                    output_teacher.div_(self.config['temperature']).softmax(-1),
                    reduction='batchmean'
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_stats["loss"].append(loss.detach())
                train_stats["t1acc"].append(calculate_accuracy(output, target))
                train_stats["t5acc"].append(calculate_accuracy(output, target, k=5))
            lr_scheduler.step()
            train_stats["loss"] = torch.stack(train_stats["loss"]).mean().item()
            train_stats["t1acc"] = torch.stack(train_stats["t1acc"]).mean().item()
            train_stats["t5acc"] = torch.stack(train_stats["t5acc"]).mean().item()

            val_stats = self._evaluate(val_dataloader)
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                self.wandb_run.save(filepath)

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> dict:
        """
        For computing validation accuracy during training
        """
        self.model.eval()
        normalize = datasets.get_normalization(dataloader.dataset).to(self.device)
        stats = AverageMeter()
        for batch in dataloader:
            data, target = batch["image"].to(self.device), batch["target"].to(self.device)
            data = normalize(data)
            with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                output = self.model(data).float()
                loss = self.criterion(output, target).mean()
            stats.update(
                data.size(0),
                loss=loss.detach(),
                t1acc=calculate_accuracy(output, target),
                t5acc=calculate_accuracy(output, target, k=5),
            )
        stats = stats.get_average()
        return stats

    @torch.no_grad()
    def predict_batch(self, input: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
            return self.model(input).float()

    @torch.no_grad()
    def inference(self, dataset: Dataset, train=False) -> dict:
        self.model.train(train)
        data_loader = self.get_dataloader(dataset, train=False)
        
        normalize = datasets.get_normalization(dataset).to(self.device)
        results = defaultdict(list)
        # import tqdm as tqdm
        for batch in data_loader:
            # image = batch["image"]
            image = batch.pop("image")
            image = normalize(image.to(self.device))
            output = self.predict_batch(image)
            results['logits'].append(output)
            for k, v in batch.items():
                results[k].append(v)
        results = {k: torch.cat(v, dim=0).cpu() for k, v in results.items()}
        return results

    @torch.no_grad()
    def filter_noisy(self, dataset):
        self.model.eval()
        dataloader = self.get_dataloader(dataset, train=False)
        score_list = []
        is_noisy_list = []
        for batch in dataloader:
            data, target = batch["image"].to(self.device), batch["target"].to(self.device)
            target_gt = batch["target_gt"].to(self.device)
            losses = []
            for _ in range(10):
                data += torch.randn_like(data).mul_(0.1)
                output = self.model(data)
                losses.append(self.criterion(output, target))
            std, mean = torch.std_mean(torch.cat(losses, dim=-1), dim=-1)
            score = std
            is_noisy = (target != target_gt)

            score_list.append(score)
            is_noisy_list.append(is_noisy)

        score = torch.cat(score_list, dim=0).cpu()
        is_noisy = torch.cat(is_noisy_list, dim=0).cpu()
        return {
            'score': score,
            'is_noisy': is_noisy,
        }

    def distill_online_ablation1(self, train_dataset, val_dataset):
        '''
        on-line distillation ablation
        autoaugment removed, and input to both teach and student are same (RandomCrop)
        This ablation is to show that KD alone is not good enough for noisy labels
        '''
        train_dataset.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(np.array(x)).permute(2,0,1)),
        ]) # output is a (3, 32, 32) uint8 tensor
        transform_train = transforms.Compose([
            transforms_v2.RandomCrop(32, padding=4),
            transforms_v2.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(f'checkpoints',
                                  type='model',
                                  metadata=self.wandb_run.config['model'])

        for epoch in tqdm.trange(self.config["max_epoch"]):
            train_stats = {
                "loss": [],
                "t1acc": [],
                "t5acc": [],
            }
            self.model.train()
            for batch in train_dataloader:
                data, target = batch["image"].to(self.device), batch["target"].to(self.device)
                data = transform_train(data)
                self.model.train()
                output = self.model(data)
                with torch.no_grad():
                    self.model.eval()
                    output_teacher = self.model(data)
                ce_loss = self.criterion(output, target).mean()
                kl_loss = (self.config['temperature'] ** 2) * F.kl_div(
                    output.div(self.config['temperature']).log_softmax(-1),
                    output_teacher.div_(self.config['temperature']).softmax(-1),
                    reduction='batchmean'
                    )
                loss = (ce_loss + kl_loss)*0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_stats["loss"].append(loss.detach())
                train_stats["t1acc"].append(calculate_accuracy(output, target))
                train_stats["t5acc"].append(calculate_accuracy(output, target, k=5))
            lr_scheduler.step()
            train_stats["loss"] = torch.stack(train_stats["loss"]).mean().item()
            train_stats["t1acc"] = torch.stack(train_stats["t1acc"]).mean().item()
            train_stats["t5acc"] = torch.stack(train_stats["t5acc"]).mean().item()

            val_stats = self._evaluate(val_dataloader)
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def distill_online_ablation2(self, train_dataset, val_dataset):
        '''
        on-line distillation ablation
        augmentation policies swapped
        '''
        train_dataset.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(np.array(x)).permute(2,0,1)),
        ]) # output is a (3, 32, 32) uint8 tensor
        transform_student = transforms.Compose([
            transforms_v2.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])
        transform_teacher = transforms.Compose([
            transforms_v2.RandomCrop(32, padding=4),
            transforms_v2.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(f'checkpoints',
                                  type='model',
                                  metadata=self.wandb_run.config['model'])

        for epoch in tqdm.trange(self.config["max_epoch"]):
            train_stats = {
                "loss": [],
                "t1acc": [],
                "t5acc": [],
            }
            self.model.train()
            for batch in train_dataloader:
                data, target = batch["image"].to(self.device), batch["target"].to(self.device)
                output = self.model(transform_student(data))
                with torch.no_grad():
                    output_teacher = self.model(transform_teacher(data))
                ce_loss = self.criterion(output, target).mean()
                kl_loss = (self.config['temperature'] ** 2) * F.kl_div(
                    output.div(self.config['temperature']).log_softmax(-1),
                    output_teacher.div_(self.config['temperature']).softmax(-1),
                    reduction='batchmean'
                    )
                loss = (ce_loss + kl_loss)*0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_stats["loss"].append(loss.detach())
                train_stats["t1acc"].append(calculate_accuracy(output, target))
                train_stats["t5acc"].append(calculate_accuracy(output, target, k=5))
            lr_scheduler.step()
            train_stats["loss"] = torch.stack(train_stats["loss"]).mean().item()
            train_stats["t1acc"] = torch.stack(train_stats["t1acc"]).mean().item()
            train_stats["t5acc"] = torch.stack(train_stats["t5acc"]).mean().item()

            val_stats = self._evaluate(val_dataloader)
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def distill_online_ablation3(self, train_dataset, val_dataset):
        '''
        on-line distillation ablation
        augmentation policies identical
        '''
        train_dataset.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(np.array(x)).permute(2,0,1)),
        ]) # output is a (3, 32, 32) uint8 tensor
        # transform = transforms.Compose([
        #     transforms_v2.RandomCrop(32, padding=4),
        #     transforms_v2.RandomHorizontalFlip(),
        #     transforms.Lambda(lambda x: x/255.0),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        # ])
        transform = transforms.Compose([
            transforms_v2.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(f'checkpoints',
                                  type='model',
                                  metadata=self.wandb_run.config['model'])

        for epoch in tqdm.trange(self.config["max_epoch"]):
            train_stats = {
                "loss": [],
                "t1acc": [],
                "t5acc": [],
            }
            self.model.train()
            for batch in train_dataloader:
                data, target = batch["image"].to(self.device), batch["target"].to(self.device)
                output = self.model(transform(data))
                with torch.no_grad():
                    output_teacher = self.model(transform(data))
                ce_loss = self.criterion(output, target).mean()
                kl_loss = (self.config['temperature'] ** 2) * F.kl_div(
                    output.div(self.config['temperature']).log_softmax(-1),
                    output_teacher.div_(self.config['temperature']).softmax(-1),
                    reduction='batchmean'
                    )
                loss = (ce_loss + kl_loss)*0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_stats["loss"].append(loss.detach())
                train_stats["t1acc"].append(calculate_accuracy(output, target))
                train_stats["t5acc"].append(calculate_accuracy(output, target, k=5))
            lr_scheduler.step()
            train_stats["loss"] = torch.stack(train_stats["loss"]).mean().item()
            train_stats["t1acc"] = torch.stack(train_stats["t1acc"]).mean().item()
            train_stats["t5acc"] = torch.stack(train_stats["t5acc"]).mean().item()

            val_stats = self._evaluate(val_dataloader)
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)

    def fit_nrosd_ablation1(self, train_dataset: Dataset, val_dataset: Dataset):
        train_dataset.transform = self.get_transform(self.config['student_aug'], train_dataset)
        train_dataset.transform2 = self.get_transform(self.config['teacher_aug'], train_dataset)

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        self.model = torch.compile(self.model)
        # self.model = torch.jit.script(self.model)
        dp_model = nn.DataParallel(self.model)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config["enable_amp"])
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(f'checkpoints',
                                  type='model',
                                  metadata=self.wandb_run.config['model'])

        distill_criterion = self.get_distill_loss_fn(
                                            self.config["distill_loss_fn"],
                                            self.config['temperature']
                                            )
        alpha = self.config['alpha']

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True, position=0):
            train_stats = {
                "loss": [],
                "t1acc": [],
                "t5acc": [],
                "target_loss": [],
                "distill_loss": [],
            }
            self.model.train()
            for batch in tqdm.tqdm(train_dataloader, desc=f'Ep {epoch}', dynamic_ncols=True, leave=False, position=1):
                target = batch["target"].to(self.device)
                data, data2 = batch["image"], batch['image2']
                with torch.cuda.amp.autocast(enabled=self.config["enable_amp"]):
                    output = dp_model(data)
                    output_teacher = dp_model(data2)
                    target_loss = self.criterion(output, target).mean()
                    distill_loss = distill_criterion(output, output_teacher).mean()
                    loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_stats["loss"].append(loss.detach())
                train_stats["t1acc"].append(calculate_accuracy(output, target))
                train_stats["t5acc"].append(calculate_accuracy(output, target, k=5))
                train_stats["target_loss"].append(target_loss.detach())
                train_stats["distill_loss"].append(distill_loss.detach())
            lr_scheduler.step()
            train_stats["loss"] = torch.stack(train_stats["loss"]).mean().item()
            train_stats["t1acc"] = torch.stack(train_stats["t1acc"]).mean().item()
            train_stats["t5acc"] = torch.stack(train_stats["t5acc"]).mean().item()
            train_stats["target_loss"] = torch.stack(train_stats["target_loss"]).mean().item()
            train_stats["distill_loss"] = torch.stack(train_stats["distill_loss"]).mean().item()

            val_stats = self._evaluate(val_dataloader)
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t1acc']:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    **{'train_'+k: v for k, v in train_stats.items()},
                    **{'val_'+k: v for k, v in val_stats.items()},
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                artifact.add_file(filepath)
        self.wandb_run.log_artifact(artifact)



@torch.no_grad()
def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor, k=1):
    """Computes top-k accuracy"""
    k = min(k, pred.size(-1)) # in case num_classes is smaller than k.
    pred = torch.topk(pred, k, -1).indices
    correct = pred.eq(target[..., None].expand_as(pred)).any(dim=-1)
    accuracy = correct.float().mean().mul(100.0)
    return accuracy


class AverageMeter:
    def __init__(self):
        self.cnt = 0
        self.stats = defaultdict(int)

    @torch.no_grad()
    def update(self, size: int, **kwargs):
        """
        size: batch size
        kwargs: key-value pairs of stats to update
        """
        a = self.cnt / (self.cnt + size)
        b = size / (self.cnt + size)
        for key, value in kwargs.items():
            self.stats[key] = a*self.stats[key] + b*value
        self.cnt += size

    def get_average(self) -> dict:
        return self.stats


class TabularEMA(nn.Module):
    def __init__(self, num_samples: int, num_classes: int, beta: float = 0.9999):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.beta = beta
        self.register_buffer('table', torch.ones((num_samples, num_classes)).div_(self.num_classes))

    def forward(self, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        self.table[indices].mul_(self.beta).add_(input, alpha=1-self.beta)
        return self.table[indices]