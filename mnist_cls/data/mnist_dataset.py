import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple


class MNISTDataModule:
    def __init__(
        self,
        data_dir: str = './data/mnist',
        batch_size: int = 128,
        num_workers: int = 4,
        normalize: bool = True,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.augment = augment
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self):
        transform_list = [transforms.ToTensor()]
        train_transform_list = transform_list.copy()

        if self.augment:
            train_transform_list = [
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
            ] + train_transform_list

        if self.normalize:
            normalize_transform = transforms.Normalize((0.1307,), (0.3081,))
            transform_list.append(normalize_transform)
            train_transform_list.append(normalize_transform)

        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose(transform_list)

        train_dataset = datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )

        train_dataset, val_dataset = random_split(
            train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if self.train_loader is None:
            self.setup()
        return self.train_loader, self.val_loader, self.test_loader
