import torch
import torchvision
import numpy as np
from typing import Tuple, Any
from PIL import Image
import os

class Clothing1M(torchvision.datsets.ImageFolder):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None
        split: bool = ''
        ) -> None: # No download, noise_type
        """
            Directory structure is different from Clothing1M official version..
            [Our directory structure]
            Clothing1M/
            ├── clean_train
            │ ├── 0
            │ ├── ⋮
            │ └── 13
            ├── clean_test
            │ ├── 0
            │ ├── ⋮
            │ └── 13
            ├── clean_val
            │ ├── 0
            │ ├── ⋮
            │ └── 13
            └──noisy_train
            ├── 0
            ├── ⋮
            └── 13

            [Args]

            train   bool        True    : getitem from train set
                                False   : getitem from test / val set

            transform           pytorch transform

            target_transform    pytorch target transform

            split   str         clean       : clean_train if train is True,  else NotImplementedError
                                ''(default) : noisy_train if train is True,  else clean_test
                                val         : clean_val   if train is False, else NotImplementedError
        """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None
        split: bool = ''
        ) -> None: # No download, noise_type
        # Adapted from https://github.com/Yikai-Wang/SPR-LNL/blob/af29e431bf7bd4d7840ce87c08b2ae9ebafeb7d1/dataset.py#L331
        # This implementation does not support cutmix mode / label updating.
        # Animal-10N ftp 들어가보면 raw_img ver과 binary ver 있는데, raw_img ver이 맞는것같다.(label을 path.split으로 정하는것을 보니)
        self.train = train
        if self.train:
            if split in ['', 'noisy']:
                split = 'noisy_train'
            elif split == 'clean':
                split = 'clean_train'
            else:
                raise NotImplementedError
        else:
            if split in ['', 'test']:
                split = 'clean_test'
            elif split == 'val':
                split = 'clean_val'
            else:
                raise NotImplementedError

        root = os.path.join(root, split)
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform
                         )

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return {
            'image': image,
            'target': target
        }# No target_gt: real-world noisy dataset.


if __name__ == '__main__':
    import pdb
    root = './datasets/Clothing1M'
    animal_dset = Animal10(root, train=True)
    pdb.set_trace()
