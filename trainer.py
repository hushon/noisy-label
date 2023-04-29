import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm.auto as tqdm
import os
import wandb
from wandb.sdk.wandb_run import Run
import pdb
from models import MeanAbsoluteError


class Trainer:
    def __init__(self, model: nn.Module, config: dict, wandb_run: Run =None):
        self.model = model.cuda()
        self.config = config
        self.wandb_run = wandb_run
        self.criterion = self.get_loss_fn().cuda()

    def get_optimizer(self,
                      model: nn.Module
                      ) -> torch.optim.Optimizer:
        if self.config["optimizer"] == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config["init_lr"],
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config["init_lr"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise NotImplementedError

    def get_lr_scheduler(self,
                         optimizer: torch.optim.Optimizer
                         ) -> torch.optim.lr_scheduler.LRScheduler:
        if self.config["lr_scheduler"] == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config["max_epoch"],
                eta_min=0.0,
            )
        elif self.config["lr_scheduler"] == "multistep":
            n = self.config["max_epoch"] // 10
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[3*n, 6*n, 8*n],
                gamma=0.2,
            )
        else:
            raise NotImplementedError

    def get_loss_fn(self) -> torch.nn.Module:
        if self.config["loss_fn"] == "cross_entropy":
            return nn.CrossEntropyLoss(reduction="none")
        if self.config["loss_fn"] == "mae":
            return MeanAbsoluteError(reduction="none")
        else:
            raise NotImplementedError

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

    def fit(self, train_dataset, val_dataset):
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
                data, target = batch["image"].cuda(), batch["target"].cuda()
                output = self.model(data)
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
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['t1acc']:.2f}, Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['t5acc']:.2f}")
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
    def _evaluate(self,
                  dataloader: torch.utils.data.DataLoader
                  ) -> dict:
        self.model.eval()
        stats = {
            "loss": [],
            "t1acc": [],
            "t5acc": [],
        }
        for batch in dataloader:
            data, target = batch["image"].cuda(), batch["target"].cuda()
            output = self.model(data)
            loss = self.criterion(output, target).mean()
            stats["loss"].append(loss.detach())
            stats["t1acc"].append(calculate_accuracy(output, target))
            stats["t5acc"].append(calculate_accuracy(output, target, k=5))
        stats["loss"] = torch.stack(stats["loss"]).mean().item()
        stats["t1acc"] = torch.stack(stats["t1acc"]).mean().item()
        stats["t5acc"] = torch.stack(stats["t5acc"]).mean().item()
        return stats

    @torch.no_grad()
    def filter_noisy(self, dataset):
        self.model.eval()
        dataloader = self.get_dataloader(dataset, train=False)
        score_list = []
        is_noisy_list = []
        for batch in dataloader:
            data, target = batch["image"].cuda(), batch["target"].cuda()
            target_gt = batch["target_gt"].cuda()
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


@torch.no_grad()
def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, k=1):
    """Computes top-k accuracy"""
    pred = torch.topk(output, k, 1, True, True).indices
    correct = pred.eq(target[..., None].expand_as(pred)).any(dim=1)
    accuracy = correct.float().mean().mul(100.0)
    return accuracy