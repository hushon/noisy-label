import torch
import torchvision
import numpy as np
from typing import Tuple, Any
from PIL import Image
import os
from simplejpeg import decode_jpeg


def image_loader(path: str) -> Image.Image:
    try:
        with open(path, 'rb') as fp:
            image = decode_jpeg(fp.read(), colorspace='RGB')
        image = Image.fromarray(image)
    except:
        image = Image.open(path).convert('RGB')
        print(f'Resorting to PIL: {path}')
    return image


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
        transform2 = None,
        target_transform = None,
        ) -> None:
        assert split in ['noisy_train', 'clean_train', 'clean_val', 'clean_test']
        self.split = split
        super().__init__(root=os.path.join(root, 'Clothing1M-processed', self.split),
                         transform=transform,
                         target_transform=target_transform,
                         loader=image_loader
                         )
        self.transform2 = transform2

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            image = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = {
            'image': image,
            'target': target,
        }# No target_gt: real-world noisy dataset.
        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(sample),
            })
        return output
