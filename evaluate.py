
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
import datasets
from train import get_model
import pdb
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# API guide: https://docs.wandb.ai/guides/track/public-api-guide

np.random.seed(0)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Evaluating Config', add_help=False)
parser.add_argument('--config', type=str, required=True, help="./configs/eval_base.yml")
args = parser.parse_args()

def load_artifact_from_wandb(wandb_run, config):
    # artifact full name : "my-entity/my-project/artifact:alias"
    artifact = wandb_run.use_artifact(config['model_restoration']['artifact_name'],
                                      type='model')
    model_entry = artifact.get_path(f"model_{config['model_restoration']['epoch']}.pth")
    ckpt_path = model_entry.download() 
     
    return ckpt_path, artifact.metadata


def load_files_from_wandb(config):
    run_path = config['model_restoration']['file_run_path'] 
    api = wandb.Api()
    run = api.run(run_path) 
    metadata = run.config['model'] # TODO: config에서 받도록 하면 사실 api 안불러와도 됨.
    epoch = config['model_restoration']['epoch'] 
    ckpt_path = wandb.restore(f"model_{epoch}.pth",
                               run_path=run_path) # TODO: checkpoint인지 확인해보기.
    return ckpt_path.name, metadata

def get_filter_dataset(dataset, noise_rate, noise_type, random_seed):

    transform_train = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.tensor(np.array(x)).permute(2,0,1))
    ])

    transform_normalize = transforms.Compose([
        transforms.Lambda(lambda x: x/255.0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_tta = transforms.Compose([
        AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.Lambda(lambda x: x/255.0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train = datasets.NoisyCIFAR10("./data", download=True, transform=transform_train, noise_rate=noise_rate, noise_type=noise_type, random_seed=random_seed)

    return train, transform_normalize, transform_tta

def main():
    # Load YAML config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    pprint.pprint(config)

    wandb_run = wandb.init(
        **config['wandb'],
        config=config,
    )
    
    restore = config['model_restoration']['mode']
    
    if restore == 'artifact':
        ckpt_path, metadata = load_artifact_from_wandb(wandb_run, config)
    elif restore == 'file':
        # TODO: upload artifact if restore= == 'file'.
        # Is it possible?
        ckpt_path, metadata = load_files_from_wandb(config)
    else:
        raise NotImplementedError("Only artifact and file mode are supported..")

    model = get_model(**metadata).cuda()
    state_dict = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(state_dict, strict=True)

    train_dataset, transform_normalize, transform_tta = get_filter_dataset(**config["data"])
    
    trainer = Trainer(
                    model=model,
                    config=config['trainer'],
                    wandb_run=wandb_run
                    )
    
    with torch.no_grad():
        trainer.model.eval()
        dataloader = trainer.get_dataloader(train_dataset, train=False)

        result = {
            'std': [],
            'mean': [],
            'is_noisy': [],
        }
        for batch in dataloader:
            data, target = batch["image"].cuda(), batch["target"].cuda()
            target_gt = batch["target_gt"].cuda()
            losses = []
            for _ in range(10):
                # data_ = torch.normal(transform_normalize(data), 0.1)
                data_ = transform_tta(data)
                output = trainer.model(data_)
                loss = trainer.criterion(output, target)
                losses.append(loss)
            std, mean = torch.std_mean(torch.stack(losses, dim=-1), dim=-1)
            is_noisy = (target != target_gt)
            result['std'].append(std)
            result['mean'].append(mean)
            result['is_noisy'].append(is_noisy)

        result['std'] = torch.cat(result['std'], dim=0).cpu()
        result['mean'] = torch.cat(result['mean'], dim=0).cpu()
        result['is_noisy'] = torch.cat(result['is_noisy'], dim=0).cpu()

    score = result['mean']
    # score = result['mean'] + 0.5*result['std']
    # score = result['mean'] - 0.5*result['std']
    # score = result['std']

    blue_data = score[~result['is_noisy']]
    red_data = score[result['is_noisy']]

    plt.hist([blue_data, red_data], color=['blue', 'red'], label=["clean", "noisy"])
    # plt.hist(blue_data, bins=np.arange(0, 8, 0.5), color='blue', alpha=0.5, label="clean")
    # plt.hist(red_data, bins=np.arange(0, 8, 0.5), color='red', alpha=0.5, label="noisy")

    # add title and axis labels
    plt.title("noisy label filtering")
    plt.xlabel("score")
    plt.ylabel("frequency")

    # add legend
    plt.legend()

    # TODO: 로깅 어떻게 더 이쁘게하지? 한번 run 할때 score 다양하게 뽑아서 다양하게 logging?
    # display the histogram
    wandb.log({'chart':wandb.Image(plt),
               'auroc': roc_auc_score(result['is_noisy'], score)})
    # plt.show()

    # wandb_run.alert(
    #     title="Training finished",
    #     text="this is a test message",
    #     level=wandb.AlertLevel.INFO,
    # )
    wandb_run.finish()

if __name__ == '__main__':
    main()