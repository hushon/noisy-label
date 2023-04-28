import torch
import torchvision
import numpy as np
from typing import Tuple, Any
from PIL import Image
import os

# TODO: Imagenet..?
class WebVision(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        num_class: int= 0,
        split: str = ''):
        # Adapted from https://github.com/LiJunnan1992/DivideMix/blob/master/dataloader_webvision.py
        # 'all', and 'unlabeled' mode are deprecated. We only support 'train(with label)', 'test' and 'val'.
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train
        """
            Directory structure of official version.
            In this implementation, we follow the official version but utilize only Google Images.
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
        # TODO: subset만 한다고 했을 때, 50 class sorting이 webvision train/test // imagenet train/test 다 동일한가..? metadata 열어보기.
        finfo = None
        if self.train:
            finfo = 'train_filelist_google.txt'
        else:
            if split in ['', 'test']:
                finfo = 'val_filelist.txt'
            elif split == 'val':
                finfo = 'test_filelist.txt'
            else:
                raise NotImplementedError

        # TODO: self.root naming convention across datasets.
        info_file = os.path.join(self.root, 'info', finfo)

        self.img_list, self.labels = self.get_filepath(info_file, num_class)

    def __len__(self):
            return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        target = self.labels[img_path]
        target = torch.tensor(target)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            'image': image,
            'target': target
        } # No target_gt: real-world noisy dataset.

    @staticmethod
    def get_filepath(info_file, num_class):
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

if __name__ == '__main__':
    import pdb
    root = './data/WebVision/google_low_resolution'
    dset = WebVision(root, train=True)
    pdb.set_trace()


# TODO: Or, inherit from torchvision imagenet dataset?
class ImageNetVal(torch.utils.data.Dataset):
    #This class is for validating webvision model.
    
    pass
