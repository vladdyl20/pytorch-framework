import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_framework.engine.classification import ClassificationTrainer
from pytorch_framework.utils.seed import set_seed


class FashionMNISTCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_units: int = 16,
        output_shape: int = 10,
    ):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 7 * 7,
                out_features=output_shape,
            ),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)

        return x


def main():
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="data/fashion_mnist",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root="data/fashion_mnist",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = FashionMNISTCNN(
        input_channels=1,
        hidden_units=16,
        output_shape=len(train_dataset.classes),
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=3,
    )


if __name__ == "__main__":
    main()