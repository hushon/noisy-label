import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import tqdm.auto as tqdm
import os
import wandb
from wandb.sdk.wandb_run import Run
import pdb
from models import MeanAbsoluteError, ReverseCrossEntropyLoss, SymmetricCrossEntropyLoss, \
    GeneralizedCrossEntropyLoss
from models import KLDivDistillationLoss, L1DistillationLoss, SmoothL1DistillationLoss
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2
import numpy as np


class Trainer:
    def __init__(self, model: nn.Module, config: dict, wandb_run: Run, device='cuda:0'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.wandb_run = wandb_run
        self.criterion = self.get_loss_fn(self.config["loss_fn"]).to(self.device)

    def get_optimizer(self,
                      model: nn.Module
                      ) -> torch.optim.Optimizer:
        match self.config["optimizer"]:
            case "sgd":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.config["init_lr"],
                    momentum=self.config["momentum"],
                    weight_decay=self.config["weight_decay"],
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

    def get_lr_scheduler(self,
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
            case _:
                raise NotImplementedError(self.config["lr_scheduler"])
        return lr_scheduler

    @staticmethod
    def get_loss_fn(fn_name) -> nn.Module:
        match fn_name:
            case "cross_entropy":
                fn = nn.CrossEntropyLoss(reduction="none")
            case "mae":
                fn = MeanAbsoluteError(reduction="none")
            case "reverse_cross_entropy":
                fn = ReverseCrossEntropyLoss(reduction="none")
            case "symmetric_cross_entropy":
                fn = SymmetricCrossEntropyLoss(reduction="none")
            case "generalized_cross_entropy":
                fn = GeneralizedCrossEntropyLoss(reduction="none")
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
            case _:
                raise NotImplementedError(fn_name)
        return fn

    @staticmethod
    def get_transform(op_name):
        CIFAR10_MEAN_STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        CIFAR100_MEAN_STD = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        match op_name:
            case "totensor":
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: torch.tensor(np.array(x)).permute(2,0,1).contiguous()),
                ]) # output is a (3, 32, 32) uint8 tensor
            # case "randomcrop":
            #     transform = transforms.Compose([
            #         transforms_v2.RandomCrop(32, padding=4),
            #         transforms_v2.RandomHorizontalFlip(),
            #         transforms.Lambda(lambda x: x/255.0),
            #         transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
            #     ])
            # case "autoaugment":
            #     transform = transforms.Compose([
            #         transforms_v2.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            #         transforms.Lambda(lambda x: x/255.0),
            #         transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
            #     ])

            case "randomcrop":
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Lambda(lambda x: x/255.0),
                    transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
                ])
            case "autoaugment":
                transform = transforms.Compose([
                    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                    transforms.Lambda(lambda x: x/255.0),
                    transforms.Normalize(*CIFAR10_MEAN_STD, inplace=True),
                ])
            case _:
                raise NotImplementedError(op_name)
        return transform

    def get_dataloader(self,
                       dataset: torch.utils.data.Dataset,
                       train=True
                       ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=train,
            num_workers=self.config["num_workers"],
            drop_last=train,
        )

    def fit(self, train_dataset: Dataset, val_dataset: Dataset):
        train_dataset.transform = self.get_transform('totensor')
        transform_train = self.get_transform('randomcrop')
        # transform_train = self.get_transform('autoaugment')

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        # self.model = torch.compile(self.model)
        self.model = torch.jit.script(self.model)

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
                output = self.model(transform_train(data))
                loss = self.criterion(output, target).mean()
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

    def fit_nrosd(self, train_dataset: Dataset, val_dataset: Dataset):
        train_dataset.transform = self.get_transform('totensor')
        transform_train = self.get_transform(self.config['student_aug'])
        transform_teacher = self.get_transform(self.config['teacher_aug'])

        train_dataloader = self.get_dataloader(train_dataset, train=True)
        val_dataloader = self.get_dataloader(val_dataset, train=False)

        # self.model = torch.compile(self.model)
        self.model = torch.jit.script(self.model)

        optimizer = self.get_optimizer(self.model)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        # artifact = wandb.Artifact(f'checkpoints_{self.wandb_run.id}',
        artifact = wandb.Artifact(f'checkpoints',
                                  type='model',
                                  metadata=self.wandb_run.config['model'])

        distill_criterion = self.get_distill_loss_fn(
                                            self.config["distill_loss_fn"],
                                            self.config['temperature']
                                            )
        alpha = self.config['alpha']

        for epoch in tqdm.trange(self.config["max_epoch"], dynamic_ncols=True):
            train_stats = {
                "loss": [],
                "t1acc": [],
                "t5acc": [],
                "target_loss": [],
                "distill_loss": [],
            }
            self.model.train()
            for batch in train_dataloader:
                data, target = batch["image"].to(self.device), batch["target"].to(self.device)
                output = self.model(transform_train(data))
                with torch.no_grad():
                    output_teacher = self.model(transform_teacher(data))
                target_loss = self.criterion(output, target).mean()
                distill_loss = distill_criterion(output, output_teacher).mean()
                loss = target_loss * alpha + distill_loss * (1.0-alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
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
    def _evaluate(self, dataloader: torch.utils.data.DataLoader) -> dict:
        self.model.eval()
        stats = {
            "loss": [],
            "t1acc": [],
            "t5acc": [],
        }
        total_size = 0
        for batch in dataloader:
            data, target = batch["image"].to(self.device), batch["target"].to(self.device)
            total_size += data.size(0)
            output = self.model(data)
            loss = self.criterion(output, target).mean()
            stats["loss"].append(loss.detach()*data.size(0))
            stats["t1acc"].append(calculate_accuracy(output, target)*data.size(0))
            stats["t5acc"].append(calculate_accuracy(output, target, k=5)*data.size(0))
        stats["loss"] = torch.tensor(stats["loss"]).sum().div(total_size).item()
        stats["t1acc"] = torch.tensor(stats["t1acc"]).sum().div(total_size).item()
        stats["t5acc"] = torch.tensor(stats["t5acc"]).sum().div(total_size).item()
        return stats

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


@torch.no_grad()
def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor, k=1):
    """Computes top-k accuracy"""
    k = min(k, pred.size(-1)) # in case num_classes is smaller than k.
    pred = torch.topk(pred, k, -1).indices
    correct = pred.eq(target[..., None].expand_as(pred)).any(dim=-1)
    accuracy = correct.float().mean().mul(100.0)
    return accuracy
