import torch
import torchvision
import numpy as np


class NoisyCIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR-10 Dataset with synthetic label noise."""
    num_classes = 10
    asymm_label_transition = {
        9: 1,  # truck -> automobile
        2: 0,  # bird -> airplane
        3: 5,  # cat -> dog
        5: 3,  # dog -> cat
        4: 7,  # deer -> horse
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noise_rate : float = 0.2,
            noise_type : str = "symmetric",
            random_state: int = 42,
            ) -> None:
        super().__init__(root, train=train, transform=transform,
            target_transform=target_transform, download=download)
        assert self.train == True
        assert 0.0 <= noise_rate <= 1.0
        assert noise_type in ["symmetric", "asymmetric"]

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.noise_rate = noise_rate
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

        if noise_type == "symmetric":
            self._inject_symmetric_noise()
        elif noise_type == "asymmetric":
            self._inject_asymmetric_noise()

    def _inject_symmetric_noise(self):
        self.targets_gt = self.targets.copy()
        num_noisy_samples = int(self.noise_rate * len(self))
        target_mask = np.full_like(self.targets, False)
        target_mask[:num_noisy_samples] = True
        self.rng.shuffle(target_mask)
        self.targets[target_mask] = self.rng.integers(0, self.num_classes,
                                                    size=num_noisy_samples)

    def _inject_asymmetric_noise(self):
        self.targets_gt = self.targets.copy()
        num_noisy_samples = int(self.noise_rate * len(self))
        target_mask = np.full_like(self.targets, False)
        target_mask[:num_noisy_samples] = True
        target_mask = self.rng.shuffle(target_mask)
        # change labels
        for gt, tgt in self.asymm_label_transition.items():
            self.targets[target_mask & (self.targets == gt)] = tgt

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {
            'image': img,
            'target': target,
            'target_gt': self.targets_gt[index],
        }


class NoisyCIFAR100(torchvision.datasets.CIFAR100):
    """CIFAR-100 Dataset with synthetic label noise."""
    num_classes = 100

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noise_rate : float = 0.2,
            noise_type : str = "symmetric",
            random_state: int = 42,
            ) -> None:
        super().__init__(root, train=train, transform=transform,
            target_transform=target_transform, download=download)
        assert self.train == True
        assert 0.0 <= noise_rate <= 1.0
        assert noise_type in ["symmetric", "asymmetric"]

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.noise_rate = noise_rate
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

        if noise_type == "symmetric":
            transition_matrix = self._symmetric_label_transition_matrix(self.num_classes, noise_rate)
        elif noise_type == "asymmetric":
            transition_matrix = self._asymmetric_label_transition_matrix(self.num_classes, noise_rate)
        self._inject_label_noise(transition_matrix)

    @staticmethod
    def _symmetric_label_transition_matrix(n, noise_rate):
        transition_matrix = np.full((n, n), noise_rate / (n - 1))
        np.fill_diagonal(transition_matrix, 1 - noise_rate)
        return transition_matrix

    @staticmethod
    def _asymmetric_label_transition_matrix(n, noise_rate):
        eye = np.eye(n, dtype=np.int32)
        transition_matrix = (1 - noise_rate) * eye + noise_rate * np.roll(eye, 1, axis=1)
        return transition_matrix

    def _inject_label_noise(self, transition_matrix):
        self.targets_gt = self.targets.copy()
        eye = np.eye(self.num_classes, dtype=np.int32)
        in_dist = eye[self.targets_gt] # one-hot encoding
        out_dist = in_dist @ transition_matrix
        self.targets = self.rng.choice(np.arange(len(out_dist)), p=out_dist)
        pass

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {
            'image': img,
            'target': target,
            'target_gt': self.targets_gt[index],
        }
