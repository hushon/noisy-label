import torch
import torchvision
import numpy as np
from typing import Tuple, Any
from torchvision

     
        
class Animal10(Dataset):

    def __init__(self, split='train', data_path=None, transform=None, cutmix=False):
        self.train = split == 'train'
        self.cutmix = cutmix

        data_path = data_path.lower()

        self.image_dir = os.path.join(data_path, split + 'ing')

        self.image_files = [f for f in listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]

        self.targets = []

        for path in self.image_files:
            label = path.split('_')[0]
            self.targets.append(int(label))

        self.mislabeled_targets = self.targets

        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])

        image_origin = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image_origin)
            if self.cutmix:
                image1 = self.transform(image_origin)

        label = self.targets[index]
        label = torch.from_numpy(np.array(label).astype(np.int64))

        if self.train:
            if self.cutmix:
                return image, image1, label, index
            return image, label, index
        else:
            return image, label

    def __len__(self):
        return len(self.targets)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]