import torch
import torchvision
import numpy as np
from typing import Tuple, Any
from PIL import Image
import os

class Clothing1M(torchvision.datasets.ImageFolder):
    """
        Directory structure is different from Clothing1M official version.
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

        split   str         noisy_train (default)
                            clean_train
                            clean_val
                            clean_test

        transform           pytorch transform

        target_transform    pytorch target transform
    """
    def __init__(
        self,
        root: str,
        split: str = 'noisy_train',
        transform = None,
        target_transform = None,
        ) -> None:
        assert split in ['noisy_train', 'clean_train', 'clean_val', 'clean_test']
        self.split = split
        super().__init__(root=os.path.join(root, 'Clothing1M', self.split),
                         transform=transform,
                         target_transform=target_transform
                         )

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return {
            'image': image,
            'target': target,
        }# No target_gt: real-world noisy dataset.
