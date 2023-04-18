import torch
import torch.nn as nn


class Trainer(nn.Module):
    def __init__(self, train_loader, test_loader, model):
        super(Trainer, self).__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = get_optimizer(
            model=model,
        )
        self.criterion = nn.CrossEntropyLoss()