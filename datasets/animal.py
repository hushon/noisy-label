import torch
import torchvision
import numpy as np
from typing import Tuple, Any
from PIL import Image
import os
import gdown


class Animal10N(torch.utils.data.Dataset):
    """
    Song, H., Kim, M., and Lee, J., SELFIE: Refurbishing Unclean Samples for Robust Deep Learning,
    In Proc. 36th Int'l Conf. on Machine Learning (ICML), Long Beach, California, June 2019
    """
    animal10n_url = 'https://drive.google.com/open?id=1oXacCyyCMgnnfGC2lRhDaLv6jjwTljHH'
    md5 = 'f0f2b420c3371b516728ed0ef9fb2108'
    filename = 'raw_image_ver.zip'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        download: bool = False,
        ) -> None:
        # Adapted from https://github.com/Yikai-Wang/SPR-LNL/blob/af29e431bf7bd4d7840ce87c08b2ae9ebafeb7d1/dataset.py#L331
        # This implementation does not support cutmix mode / label updating.
        # Animal-10N ftp 들어가보면 raw_img ver과 binary ver 있는데, raw_img ver이 맞는것같다.(label을 path.split으로 정하는것을 보니)
        self.train = train
        split = 'train' if self.train else 'test'

        self.root = root
        if download:
            self._download()
        self.image_dir = os.path.join(self.root, f'{split}ing')
        self.image_files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]

        self.targets = []
        for path in self.image_files:
            label = path.split('_')[0]
            self.targets.append(int(label))

        self.transform = transform
        self.target_transform = target_transform

    def _download(self):
        gdown.cached_download(
            self.animal10n_url,
            os.path.join(self.root, 'animal-10n', self.filename),
            md5=self.md5,
            postprocess=gdown.extractall,
            fuzzy=True
            )
        gdown.extractall(os.path.join(root, 'animal-10n', 'raw_image.zip'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_path)

        # target = self.targets[index]
        # target = torch.from_numpy(np.array(target).astype(np.int64)) # why?
        target = torch.tensor(self.targets[index]) # It may be same as commented above. (shape, dtype)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            'image': image,
            'target': target
        } # No target_gt: real-world noisy dataset.

if __name__ == '__main__':
    import pdb
    root = './data/Animal-10N'
    animal_dset = Animal10(root, train=True)
    pdb.set_trace()
