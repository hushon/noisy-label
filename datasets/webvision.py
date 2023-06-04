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


class WebVisionV1(torch.utils.data.Dataset):
    """
    Li, Wen, et al. "Webvision database: Visual learning and understanding
    from web data." arXiv preprint arXiv:1708.02862 (2017).

    We utilize only Google Images following Chen, Pengfei, et al.
    "Understanding and utilizing deep neural networks trained with noisy
    labels." International Conference on Machine Learning. PMLR, 2019.

    Directory structure of official version.
        WebVision/
        ├── info
        │ ├── train_filelist_google.txt
        │ ├── val_filelist.txt
        │ ├── test_filelist.txt
        │ ├── ⋮
        │ └── ...
        ├── google
        │ ├── q0001
        │ │ └── ******.jpg
        │ ├── ⋮
        │ └── q
        ├── val_images
        │ ├── val000001.jpg
        │ ├── ⋮
        │ └── val050000.jpg
        └── test_images
            ├── test000001.jpg
            ├── ⋮
            └── test050000.jpg
    """
    metafile = {
        'train': 'train_filelist_google.txt',
        'val': 'val_filelist.txt',
        'test': 'test_filelist.txt',
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        num_classes: int = 50,
        transform = None,
        transform2 = None,
        target_transform = None,
        ):
        # Adapted from https://github.com/LiJunnan1992/DivideMix/blob/master/dataloader_webvision.py
        # 'all', and 'unlabeled' mode are deprecated. We only support 'train(with label)', 'test' and 'val'.
        assert split in ['train', 'val', 'test']
        assert 1 <= num_classes <= 1000

        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform

        info_file = os.path.join(self.root, 'webvision', 'info', self.metafile[self.split])
        self.img_list, self.labels = self._get_filepath(info_file, self.num_classes)

        match self.split:
            case 'train':
                self.base_path = os.path.join(self.root, 'webvision')
            case 'val':
                self.base_path = os.path.join(self.root, 'webvision', 'val_images_256')
            case 'test':
                self.base_path = os.path.join(self.root, 'webvision', 'test_images_256')
            case _:
                raise NotImplementedError

    @staticmethod
    def _get_filepath(info_file, num_class):
        # TODO: subset만 한다고 했을 때, 50 class sorting이 webvision train/test // imagenet train/test 다 동일한가..? metadata 열어보기.
        with open(info_file) as f:
            lines=f.readlines()
        img_list = []
        labels = {}
        for line in lines:
            img, target = line.split()
            target = int(target)
            if (num_class == 0) or (target < num_class):
                img_list.append(img)
                labels[img]=target
        return img_list, labels

    def __len__(self):
            return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        image = image_loader(os.path.join(self.base_path, img_path))

        target = self.labels[img_path]
        target = torch.tensor(target)

        if self.transform is not None:
            image1 = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        output = {
            'image': image1,
            'target': target,
        }# No target_gt: real-world noisy dataset. # TODO: clean key_list..?
        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(image),
            })
        return output
