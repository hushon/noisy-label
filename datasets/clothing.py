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


# class Clothing1M(torchvision.datasets.ImageFolder):
#     """
#         Directory structure is different from Clothing1M official version.
#         [Our directory structure]
#         Clothing1M/
#         ├── clean_train
#         │ ├── 0
#         │ ├── ⋮
#         │ └── 13
#         ├── clean_test
#         │ ├── 0
#         │ ├── ⋮
#         │ └── 13
#         ├── clean_val
#         │ ├── 0
#         │ ├── ⋮
#         │ └── 13
#         └──noisy_train
#         ├── 0
#         ├── ⋮
#         └── 13

#         [Args]

#         split   str         noisy_train (default)
#                             clean_train
#                             clean_val
#                             clean_test

#         transform           pytorch transform

#         target_transform    pytorch target transform
#     """
#     def __init__(
#         self,
#         root: str,
#         split: str = 'noisy_train',
#         transform = None,
#         transform2 = None,
#         target_transform = None,
#         ) -> None:
#         assert split in ['noisy_train', 'clean_train', 'clean_val', 'clean_test']
#         self.split = split
#         super().__init__(root=os.path.join(root, 'Clothing1M-processed', self.split),
#                          transform=transform,
#                          target_transform=target_transform,
#                          loader=image_loader
#                          )
#         self.transform2 = transform2

#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             image = self.transform(sample) # TODO : image reference before assignemnt error when transform=None
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         output = {
#             'image': image,
#             'target': target,
#         }# No target_gt: real-world noisy dataset.
#         if self.transform2 is not None:
#             output.update({
#                 'image2': self.transform2(sample),
#             })
#         return output

class Clothing1M(torch.utils.data.Dataset):
    """
        Clothing1M dataset for official dataset structure.
        Adapted from https://github.com/LiJunnan1992/DivideMix/blob/master/dataloader_clothing1M.py

        [Official Dataset Structure]
        github link: https://github.com/Cysu/noisy_label

        {root}/Clothing1M/
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
        self.img_root = os.path.join(self.root, 'images')
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
            image = self.transform(sample) # TODO : image reference before assignemnt error when transform=None
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = {
            'image': image,
            'target': target,
        }# No target_gt: real-world noisy dataset. # TODO: clean key_list..?
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
                img_path = os.path.join(self.root, 'images', l[7:]) # TODO: better code..bb
                img_list.append(img_path)
        return img_list

    def _get_label_dict(self):
        label_mode = self.split.split("_")[0]
        assert label_mode in ["clean", "noisy"]
        label_path = os.path.join(self.root, f"{label_mode}_label_kv.txt")
        label_dict = {}
        with open(label_path, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = os.path.join(self.root, 'images', entry[0][7:])
                label_dict[img_path] = int(entry[1])
        return label_dict


from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch

class clothing_dataset(Dataset):
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        if mode == 'all':
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    train_imgs.append(img_path)
            # random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            self.class_num = class_num
            # random.shuffle(self.train_imgs)
        elif self.mode == "labeled":
            train_imgs = paths
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == "unlabeled":
            train_imgs = paths
            pred_idx = (1-pred).nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

        elif mode=='test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.test_imgs.append(img_path)
        elif mode=='val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, img_path
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)



if __name__ == "__main__":
    import pdb
    root = '/dev/shm/data'
    # TODO: Dividemix clothing1M uses different mean, std!!
    from torchvision import transforms
    transform_train = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
    ])

    noisy_train_old = Clothing1M(root, 'noisy_train', transform=transform_train) # No transform
    noisy_train_new = Clothing1MOfficial(root, 'noisy_train', transform=transform_train)
    noisy_train_off = clothing_dataset(os.path.join(root, 'Clothing1M'), transform=transform_train, mode='all', num_samples=10000000)



    pdb.set_trace()
