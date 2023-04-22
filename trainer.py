import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm.auto as tqdm
import os


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        mae = 1. - torch.gather(pred, 1, labels.view(-1, 1))
        mae = mae.squeeze(1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        if self.reduction == "mean":
            return mae.mean()
        elif self.reduction == "sum":
            return mae.sum()
        elif self.reduction == "none":
            return mae
        else:
            raise NotImplementedError


class Trainer:
    def __init__(self, model, config, wandb_run=None):
        self.model = model.cuda()
        self.config = config
        self.wandb_run = wandb_run
        self.criterion = self.get_loss_fn().cuda()

    def get_optimizer(self, model) -> torch.optim.Optimizer:
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

    def get_lr_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
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

    def get_dataloader(self, dataset, train=True):
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

        for epoch in tqdm.trange(self.config["max_epoch"]):
            train_loss = []
            train_acc = []
            self.model.train()
            for batch in train_dataloader:
                data, target = batch["image"].cuda(), batch["target"].cuda()
                output = self.model(data)
                loss = self.criterion(output, target).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss)
                train_acc.append(calculate_accuracy(output, target))
            lr_scheduler.step()
            train_loss = torch.stack(train_loss).mean().item()
            train_acc = torch.stack(train_acc).mean().item()

            val_loss, val_acc = self._evaluate(val_dataloader)
            tqdm.tqdm.write(f"Ep {epoch}\tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}")
            self.wandb_run.log(
                {
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            if self.config["save_model"] and (epoch+1)%10 == 0:
                filepath = os.path.join(self.wandb_run.dir, f"model_{epoch}.pth")
                torch.save(self.model.state_dict(), filepath)
                self.wandb_run.save(filepath)

    @torch.no_grad()
    def _evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        correct = 0
        for batch in dataloader:
            data, target = batch["image"].cuda(), batch["target"].cuda()
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
            loss += self.criterion(output, target).sum()
            correct += pred.eq(target.view_as(pred)).sum()
        loss /= len(dataloader.dataset)
        accuracy = 100. * correct / len(dataloader.dataset)
        return loss.item(), accuracy.item()

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
def calculate_accuracy(output, target, k=1):
    """Computes top-k accuracy"""
    pred = torch.topk(output, k, 1, True, True).indices
    correct = pred.eq(target[..., None].expand_as(pred)).any(dim=1)
    accuracy = correct.float().mean().mul(100.0)
    return accuracy