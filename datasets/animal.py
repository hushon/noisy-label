import torch
import torchvision
import numpy as np
from typing import Tuple, Any
from PIL import Image
import os
import gdown
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


class Animal10N(torch.utils.data.Dataset):
    """
    Song, H., Kim, M., and Lee, J., SELFIE: Refurbishing Unclean Samples for Robust Deep Learning,
    In Proc. 36th Int'l Conf. on Machine Learning (ICML), Long Beach, California, June 2019

    Adapted from https://github.com/Yikai-Wang/SPR-LNL/blob/af29e431bf7bd4d7840ce87c08b2ae9ebafeb7d1/dataset.py#L331
    This implementation does not support cutmix mode / label updating.
    Animal-10N ftp 들어가보면 raw_img ver과 binary ver 있는데, raw_img ver이 맞는것같다.(label을 path.split으로 정하는것을 보니)
    """
    animal10n_url = 'https://drive.google.com/open?id=1oXacCyyCMgnnfGC2lRhDaLv6jjwTljHH'
    md5 = 'f0f2b420c3371b516728ed0ef9fb2108'
    filename = 'raw_image_ver.zip'
    base_folder = 'Animal-10N'

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            transform2 = None,
            target_transform = None,
            download: bool = False,
        ) -> None:
        self.train = train
        self.root = root
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform

        if download:
            self._download()

        if self.train:
            self.image_dir = os.path.join(self.root, self.base_folder, f'training')
        else:
            self.image_dir = os.path.join(self.root, self.base_folder, f'testing')

        self.image_files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]
        self.targets = np.array([int(path.split('_')[0]) for path in self.image_files])

    def _download(self):
        filepath = os.path.join(self.root, self.base_folder, self.filename)
        if os.path.exists(filepath):
            if gdown.md5sum(filepath) == self.md5:
                print('Files already downloaded and verified')
                return
        gdown.download(self.animal10n_url, filepath, fuzzy=True)
        gdown.extractall(filepath)
        gdown.extractall(os.path.join(self.root, self.base_folder, 'raw_image.zip'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = image_loader(image_path)
        target = self.targets[index]

        if self.transform is not None:
            image1 = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        output = {
            'image': image1,
            'target': target,
        } # No target_gt: real-world noisy dataset.

        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(image),
            })
        return output