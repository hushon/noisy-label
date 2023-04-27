import torch
import torchvision
import numpy as np
from typing import Tuple, Any


class Food101(torchvision.datasets.Food101):
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return{
            "image": img,
            "target": target,
        }