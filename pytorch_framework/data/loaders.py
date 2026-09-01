import torch
from torch.utils.data import DataLoader

from pytorch_framework.data.datasets import CustomImageDataset


def create_dataloaders(
    train_dir,
    test_dir,
    train_transform,
    test_transform,
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_dataset = CustomImageDataset(
        root_dir=train_dir,
        transform=train_transform,
    )

    test_dataset = CustomImageDataset(
        root_dir=test_dir,
        transform=test_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, train_dataset.classes