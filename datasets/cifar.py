import torch
import torchvision
import numpy as np
from typing import Tuple, Any
import os
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from PIL import Image


class Categorical(torch.distributions.Categorical):
    """Categorical distribution with specified random number generator."""
    def sample(self, sample_shape=torch.Size(), generator=None):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True, generator=generator).T
        return samples_2d.reshape(self._extended_shape(sample_shape))


class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return {
            "image": img,
            "target": target,
        }


class CIFAR100(torchvision.datasets.CIFAR100):
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return {
            "image": img,
            "target": target,
        }


# class NoisyCIFAR10(torchvision.datasets.CIFAR10):
#     """CIFAR-10 Dataset with synthetic label noise."""
#     num_classes = 10
#     asymm_label_transition = {
#         9: 1,  # truck -> automobile
#         2: 0,  # bird -> airplane
#         3: 5,  # cat -> dog
#         5: 3,  # dog -> cat
#         4: 7,  # deer -> horse
#     }

#     def __init__(
#             self,
#             root: str,
#             train: bool = True,
#             transform = None,
#             target_transform = None,
#             download: bool = False,
#             noise_rate : float = 0.2,
#             noise_type : str = "symmetric",
#             random_state: int = 42,
#             ) -> None:
#         super().__init__(root, train=train, transform=transform,
#             target_transform=target_transform, download=download)
#         assert self.train == True
#         assert 0.0 <= noise_rate <= 1.0
#         assert noise_type in ["symmetric", "asymmetric"]

#         self.data = np.array(self.data)
#         self.targets = np.array(self.targets)
#         self.noise_rate = noise_rate
#         self.random_state = random_state
#         self.rng = np.random.default_rng(self.random_state)

#         if noise_type == "symmetric":
#             self._inject_symmetric_noise()
#         elif noise_type == "asymmetric":
#             self._inject_asymmetric_noise()

#     def _inject_symmetric_noise(self):
#         self.targets_gt = self.targets.copy()
#         num_noisy_samples = int(self.noise_rate * len(self))
#         target_mask = np.full_like(self.targets, False)
#         target_mask[:num_noisy_samples] = True
#         self.rng.shuffle(target_mask)
#         self.targets[target_mask] = self.rng.integers(0, self.num_classes,
#                                                     size=num_noisy_samples)

#     def _inject_asymmetric_noise(self):
#         self.targets_gt = self.targets.copy()
#         num_noisy_samples = int(self.noise_rate * len(self))
#         target_mask = np.full_like(self.targets, False)
#         target_mask[:num_noisy_samples] = True
#         target_mask = self.rng.shuffle(target_mask)
#         # change labels
#         for gt, tgt in self.asymm_label_transition.items():
#             self.targets[target_mask & (self.targets == gt)] = tgt

#     def __getitem__(self, index):
#         img, target = super().__getitem__(index)
        # return {
        #     'image': img,
        #     'target': target,
        #     'target_gt': self.targets_gt[index],
        # }


class NoisyCIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR-10 Dataset with synthetic label noise."""

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            transform2 = None,
            target_transform = None,
            download: bool = False,
            noise_rate: float = 0.2,
            noise_type: str = "symmetric",
            random_seed: int = 42,
            ) -> None:
        super().__init__(root, train=train, transform=transform,
            target_transform=target_transform, download=download)
        assert self.train == True
        assert 0.0 <= noise_rate <= 1.0
        assert noise_type in ["symmetric", "asymmetric"]

        self.transform2 = transform2
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.noise_rate = noise_rate
        self.rng = torch.Generator().manual_seed(random_seed)

        if noise_type == "symmetric":
            self.transition_matrix = self._symmetric_transition_matrix(noise_rate)
        elif noise_type == "asymmetric":
            self.transition_matrix = self._asymmetric_transition_matrix(noise_rate)
        self._inject_label_noise(self.transition_matrix)

    @staticmethod
    def _symmetric_transition_matrix(noise_rate) -> np.ndarray:
        transition_matrix = np.full((10, 10), noise_rate / (10 - 1))
        np.fill_diagonal(transition_matrix, 1.0 - noise_rate)
        return transition_matrix

    @staticmethod
    def _asymmetric_transition_matrix(noise_rate) -> np.ndarray:
        cifar10_asymm_label_transition = {
            9: 1,  # truck -> automobile
            2: 0,  # bird -> airplane
            3: 5,  # cat -> dog
            5: 3,  # dog -> cat
            4: 7,  # deer -> horse
        }
        transition_matrix = np.eye(10)
        for k, v in cifar10_asymm_label_transition.items():
            transition_matrix[k, k] = 1.0 - noise_rate
            transition_matrix[k, v] = noise_rate
        return transition_matrix

    def _inject_label_noise(self, transition_matrix):
        self.targets_gt = self.targets.copy()
        out_dist = transition_matrix[self.targets_gt]
        self.targets = Categorical(torch.tensor(out_dist)).sample(generator=self.rng).numpy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        output = {
            'image': img1,
            'target': target,
            'target_gt': self.targets_gt[index],
        }
        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(img),
            })
        return output


class NoisyCIFAR100(torchvision.datasets.CIFAR100):
    """CIFAR-100 Dataset with synthetic label noise."""

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            transform2 = None,
            target_transform = None,
            download: bool = False,
            noise_rate: float = 0.2,
            noise_type: str = "symmetric",
            random_seed: int = 42,
            ) -> None:
        super().__init__(root, train=train, transform=transform,
            target_transform=target_transform, download=download)
        assert self.train == True
        assert 0.0 <= noise_rate <= 1.0
        assert noise_type in ["symmetric", "asymmetric"]

        self.transform2 = transform2
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.noise_rate = noise_rate
        self.rng = torch.Generator().manual_seed(random_seed)

        if noise_type == "symmetric":
            self.transition_matrix = self._symmetric_transition_matrix(noise_rate)
        elif noise_type == "asymmetric":
            self.transition_matrix = self._asymmetric_transition_matrix(noise_rate)
        self._inject_label_noise(self.transition_matrix)

    @staticmethod
    def _symmetric_transition_matrix(noise_rate) -> np.ndarray:
        transition_matrix = np.full((100, 100), noise_rate / (100 - 1))
        np.fill_diagonal(transition_matrix, 1 - noise_rate)
        return transition_matrix

    @staticmethod
    def _asymmetric_transition_matrix(noise_rate) -> np.ndarray:
        eye = np.eye(100, dtype=np.int32)
        transition_matrix = (1 - noise_rate) * eye + noise_rate * np.roll(eye, 1, axis=1)
        return transition_matrix

    def _inject_label_noise(self, transition_matrix):
        self.targets_gt = self.targets.copy()
        out_dist = transition_matrix[self.targets_gt]
        self.targets = Categorical(torch.tensor(out_dist)).sample(generator=self.rng).numpy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        output = {
            'image': img1,
            'target': target,
            'target_gt': self.targets_gt[index],
        }
        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(img),
            })
        return output


class CIFAR10N(torchvision.datasets.CIFAR10):
    """
    Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations. (ICLR2022)
    Re-annotation of the CIFAR-10/100 data which contains real-world human annotation errors.
    """
    cifarn_url = 'http://www.yliuu.com/web-cifarN/files/CIFAR-N-1.zip'
    cifarn_md5 = '666bf3cff3a944c245f2b6f62af4b919'
    cifar10n_filename = 'CIFAR-10_human.pt'

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            transform2 = None,
            target_transform = None,
            download: bool = False,
            noise_type: str = "worse_label",
            ) -> None:
        super().__init__(root, train=train, transform=transform,
            target_transform=target_transform, download=download)
        assert noise_type in ['clean_label', 'worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']
        self.noise_type = noise_type
        self.transform2 = transform2

        if download:
            if check_integrity(os.path.join(self.root, 'CIFAR-N-1.zip'), self.cifarn_md5):
                print('Files already downloaded and verified')
            else:
                download_and_extract_archive(self.cifarn_url, root, md5=self.cifarn_md5)

        noise_file = torch.load(os.path.join(self.root, 'CIFAR-N', self.cifar10n_filename), map_location=torch.device('cpu'))
        self.targets_gt = noise_file['clean_label']
        self.targets = noise_file[self.noise_type]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        output = {
            'image': img1,
            'target': target,
            'target_gt': self.targets_gt[index],
        }
        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(img),
            })
        return output


class CIFAR100N(torchvision.datasets.CIFAR100):
    """
    Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations. (ICLR2022)
    Re-annotation of the CIFAR-10/100 data which contains real-world human annotation errors.
    """
    cifarn_url = 'http://www.yliuu.com/web-cifarN/files/CIFAR-N-1.zip'
    cifarn_md5 = '666bf3cff3a944c245f2b6f62af4b919'
    cifar100n_filename = 'CIFAR-100_human.pt'

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            transform2 = None,
            target_transform = None,
            download: bool = False,
            ) -> None:
        super().__init__(root, train=train, transform=transform,
            target_transform=target_transform, download=download)
        self.transform2 = transform2

        if download:
            if check_integrity(os.path.join(self.root, 'CIFAR-N-1.zip'), self.cifarn_md5):
                print('Files already downloaded and verified')
            else:
                download_and_extract_archive(self.cifarn_url, root, md5=self.cifarn_md5)

        noise_file = torch.load(os.path.join(self.root, 'CIFAR-N', self.cifar100n_filename), map_location=torch.device('cpu'))
        self.targets_gt = noise_file['clean_label']
        self.targets = noise_file['noisy_label']
        self.clean_coarse_label = noise_file['clean_coarse_label']
        self.noisy_coarse_label = noise_file['noisy_coarse_label']

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        output = {
            'image': img1,
            'target': target,
            'target_gt': self.targets_gt[index],
        }
        if self.transform2 is not None:
            output.update({
                'image2': self.transform2(img),
            })
        return output


class NoisyCIFAR3(torchvision.datasets.CIFAR10):
    """
    CIFAR-3 Dataset with synthetic label noise.
    Symmetric noise only. For visualization purposes.
    """
    cifar3_classes = {
        'animal': ['deer', 'dog', 'horse'],
        'vehicle': ['automobile', 'ship', 'truck'],
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noise_rate: float = 0.2,
            class_type: str = 'animal',
            random_seed: int = 42,
            ) -> None:
        super().__init__(root, train=train, transform=transform,
            target_transform=target_transform, download=download)
        assert 0.0 <= noise_rate <= 1.0
        assert class_type in ['animal', 'vehicle']

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.noise_rate = noise_rate
        self.class_type = class_type
        self.rng = torch.Generator().manual_seed(random_seed)

        self._filter_cifar10()
        if self.train:
            self.transition_matrix = self._symmetric_transition_matrix(noise_rate)
            self._inject_label_noise(self.transition_matrix)

    def _filter_cifar10(self):
        cifar10_classes = self.classes
        self.classes = self.cifar3_classes[self.class_type]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        class_ids = [cifar10_classes.index(c) for c in self.classes]
        filter_idx = np.isin(self.targets, class_ids)
        self.data = self.data[filter_idx]
        self.targets = self.targets[filter_idx]
        self.targets = np.array([self.class_to_idx[cifar10_classes[t]] for t in self.targets])

    @staticmethod
    def _symmetric_transition_matrix(noise_rate) -> np.ndarray:
        transition_matrix = np.full((3, 3), noise_rate / (3 - 1))
        np.fill_diagonal(transition_matrix, 1.0 - noise_rate)
        return transition_matrix

    def _inject_label_noise(self, transition_matrix):
        self.targets_gt = self.targets.copy()
        out_dist = transition_matrix[self.targets_gt]
        self.targets = Categorical(torch.tensor(out_dist)).sample(generator=self.rng).numpy()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {
            'image': img,
            'target': target,
            'target_gt': self.targets_gt[index] if self.train else target,
        }
