import torch
import torch.nn as nn


class Trainer(nn.Module):
    def __init__(self,
                train_loader,
                test_loader,
                model,
                optimizer="sgd",
                init_lr="1e-1",
                momentum=0.9,
                weight_decay=1e-4,
                max_epoch=200,
                loss_fn="cross_entropy",
                lr_scheduler="cosine",
            ):
        super(Trainer, self).__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = self.get_optimizer(
            model=model,
            optimizer=optimizer,
            init_lr=init_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            )
        self.max_epoch = max_epoch
        self.lr_scheduler = self.get_lr_scheduler(self.optimizer, lr_scheduler, max_epoch)
        self.criterion = self.get_loss_fn(loss_fn)

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
    def get_lr_scheduler(optimizer, lr_scheduler) -> torch.optim.lr_scheduler.LRScheduler:
        if lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=200,
                eta_min=0,
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

    def train(self):
        self.model.train()
        for batch in enumerate(self.train_loader):
            data, target = batch["image"].cuda(), batch["target"].cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return test_loss, accuracy
    

def get_optimizer(model):
    return torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
    )