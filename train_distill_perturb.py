
import argparse
import os
import torch
import numpy as np
import random
import wandb
from models import resnet
import yaml
from trainer import Trainer
import pprint
from torchvision import transforms
from datasets import get_dataset
from models import get_model


np.random.seed(0)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--config', type=str, required=True, help="./configs/train_base.yaml")
args = parser.parse_args()


def main():
    # Load YAML config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    pprint.pprint(config)

    wandb_run = wandb.init(
        **config['wandb'],
        config=config,
    )
    wandb_run.save(args.config)
    wandb_run.log_code()

    train_dataset, test_dataset = get_dataset(**config["data"])

    model = get_model(**config["model"]).cuda()

    trainer = Trainer(
                    model=model,
                    config=config['trainer'],
                    wandb_run=wandb_run,
                    )

    import copy
    teacher_model = copy.deepcopy(model)
    if config['wandb']['mode'] == 'online':
        # WANDB_RUN_ID = "mquy2drg" # NoisyCIFAR10(symm,0.4)-CE
        WANDB_RUN_ID = "ozf6ujt1" # NoisyCIFAR10(symm,0.5)-CE-randflip
        # WANDB_RUN_ID = "ccnf390c" # NoisyCIFAR10(symm,0.4)-MAE
        #checkpoint = wandb.restore("model_199.pth", run_path=f"siit-iitp/noisy-label/{WANDB_RUN_ID}", replace=True)
        checkpoint = wandb.restore("model_199.pth", run_path=f"seunghee1215/noisy-label/{WANDB_RUN_ID}", replace=True)
        teacher_model.load_state_dict(torch.load(checkpoint.name, map_location="cuda"))
    else:
        print("Wandb is disabled.. The teacher model is randomly initialized.")
    teacher_model.eval()

    trainer.distill_perturb(train_dataset, test_dataset, teacher_model, N_aug=config['num_aug']) # TODO: Add n_aug to the config.yml?


    # wandb_run.alert(
    #     title="Training finished",
    #     text="this is a test message",
    #     level=wandb.AlertLevel.INFO,
    # )
    wandb_run.finish()

if __name__ == '__main__':
    main()
