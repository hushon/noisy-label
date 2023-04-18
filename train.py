
import argparse
import os
import torch
import numpy as np
import random
import wandb
from models import resnet


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--max_epoch', type=int, default=100)
# data
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data_root', type=str, default='/mnt/')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
# model
parser.add_argument('--model', type=str, default='resnet18')
# optimizer
parser.add_argument('--optimizer', type=str, default="sgd", choices=["adam", "sgd"])
parser.add_argument('--init_lr', type=float, default=1e-4)
# logging
parser.add_argument('--disable_log', action='store_true', default=False)

args = parser.parse_args()

wandb.init(
    project="hellomarket",
    entity="hellomarket",
    config=vars(args),
    mode="disabled" if args.disable_log else "online",
)


device = torch.device("cuda")


train_loader, test_loader = get_dataloader(
    dataset=args.dataset,
    data_root=args.data_root,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    )

model = get_model(
    model_name=args.model,
    num_classes=10,
).to(device)


trainer = Trainer(args=args,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    model=model,
                    loss=HLN,
                    optimizer=optim,
                    device=device,
                    wandb=wandb)

trainer.fit()

wandb.finish()