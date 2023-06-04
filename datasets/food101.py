import torch
import torchvision
import numpy as np
from typing import Tuple, Any
from PIL import Image
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


class Food101(torchvision.datasets.Food101):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform = None,
        transform2 = None,
        target_transform = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.transform2 = transform2

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = image_loader(image_file)

        if self.transform is not None:
            image1 = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        output = {
            'image': image1,
            'target': label,
        }

        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(image),
            })
        return output
