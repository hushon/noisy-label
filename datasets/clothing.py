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

class Clothing1MOfficial(torch.utils.data.Dataset):
    """
        Clothing1M dataset for official dataset structure.
        Adapted from https://github.com/LiJunnan1992/DivideMix/blob/master/dataloader_clothing1M.py

        [Official Dataset Structure]
        github link: https://github.com/Cysu/noisy_label

        {dataset_root}/clothing1M/
        ├── category_names_chn.txt
        ├── category_names_eng.txt
        ├── clean_label_kv.txt
        ├── clean_test_key_list.txt
        ├── clean_train_key_list.txt
        ├── clean_val_key_list.txt
        ├── images
        │   ├── 0
        │   ├── ⋮
        │   └── 9
        ├── noisy_label_kv.txt
        ├── noisy_train_key_list.txt
        ├── README.md
        └── venn.png

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
        super().__init__()
        assert split in ['noisy_train', 'clean_train', 'clean_val', 'clean_test']
        self.split = split

        self.root = os.path.join(root, 'Clothing1M') # TODO: check them
        self.transform = transform
        self.target_transform = target_transform
        self.loader = image_loader

        self.transform2 = transform2

        self.img_list = self._get_img_list()
        self.label_list = self._get_label_dict()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        path = self.img_list[index]
        target = self.label_list[path]

        sample = self.loader(path) # Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = {
            'image': image,
            'target': target,
        }# No target_gt: real-world noisy dataset. # TODO: clean key_list..? clean yes..?
        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(sample),
            })
        return output

    def _get_img_list(self):
        key_list_path = os.path.join(self.root, f"{self.split}_key_list.txt")
        img_list = []
        with open(key_list_path, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                img_path = os.path.join(self.root, l[7:]) # TODO: better code..bb
                img_list.append(img_path)
        return img_list

    def _get_label_dict(self):
        label_path = os.path.join(self.root, f"{self.split.split("_")[0]}_label_kv.txt")
        label_dict = {}
        with open(label_path, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = os.path.join(self.root, entry[0][7:])
                label_dict[img_path] = int(entry[1])
        return label_dict
