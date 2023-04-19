import torch
import torch.nn as nn
import torch.utils.data
import wandb


class Trainer(nn.Module):
    def __init__(self, model, config):
        super(Trainer, self).__init__()
        self.model = model
        self.config = config
        self.optimizer = self.get_optimizer(
            model=self.model,
            optimizer=self.config["optimizer"],
            init_lr=self.config["init_lr"],
            momentum=self.config["momentum"],
            weight_decay=self.config["weight_decay"],
            )
        self.lr_scheduler = self.get_lr_scheduler(self.optimizer, self.config["lr_scheduler"], self.config["max_epoch"])
        self.criterion = self.get_loss_fn(self.config["loss_fn"])



    @staticmethod
    def get_optimizer(model, optimizer, init_lr, momentum, weight_decay) -> torch.optim.Optimizer:
        if optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=init_lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=init_lr,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError

    @staticmethod
    def get_lr_scheduler(optimizer, lr_scheduler, max_epoch) -> torch.optim.lr_scheduler.LRScheduler:
        if lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epoch,
                eta_min=0.0,
            )
        elif lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=50,
                gamma=0.1,
            )
        else:
            raise NotImplementedError

    @staticmethod
    def get_loss_fn(loss_fn) -> torch.nn.Module:
        if loss_fn == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def fit(self, train_dataset, val_dataset=None):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            drop_last=True,
        )
        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["num_workers"],
                drop_last=False,
            )

        for epoch in range(self.config["max_epoch"]):
            self.model.train()
            for batch in enumerate(train_dataloader):
                data, target = batch["image"].cuda(), batch["target"].cuda()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            if val_dataset is not None:
                val_loss, val_acc = self._evaluate(val_dataloader)
                print(
                    f"Epoch: {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}"
                )
            else:
                print(f"Epoch: {epoch}, Train Loss: {loss.item():.4f}")

    @torch.no_grad()
    def _evaluate(self, data_loader):
        self.model.eval()
        loss = 0
        correct = 0
        for batch in enumerate(data_loader):
            data, target = batch["image"].cuda(), batch["target"].cuda()
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
            loss += self.criterion(output, target, reduction="sum")
            correct += pred.eq(target.view_as(pred)).sum()
        loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)
        return loss.item(), accuracy.item()


def get_optimizer(model):
    return torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
    )


@torch.no_grad()
def calculate_accuracy(output, target, k=1):
    """Computes top-k accuracy"""
    pred = torch.topk(output, k, 1, True, True).indices
    correct = pred.eq(target[..., None].expand_as(pred)).any(dim=1)
    accuracy = correct.float().mean().mul(100.0)
    return accuracy