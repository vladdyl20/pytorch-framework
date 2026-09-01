import argparse

import torch
from torch import nn

from pytorch_framework.data.download import download_dataset
from pytorch_framework.data.loaders import create_dataloaders
from pytorch_framework.data.transforms import get_basic_transform
from pytorch_framework.engine.classification import ClassificationTrainer
from pytorch_framework.models.tiny_vgg import TinyVGG
from pytorch_framework.utils.checkpoints import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = load_checkpoint(
        args.checkpoint,
        device=device,
    )

    config = checkpoint["model_config"]

    model = TinyVGG(
        input_channels=config["input_channels"],
        hidden_units=config["hidden_units"],
        output_shape=config["output_shape"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    dataset_path = download_dataset()

    transform = get_basic_transform(
        image_size=config["image_size"]
    )

    train_loader, test_loader, class_names = create_dataloaders(
        train_dir=dataset_path / "train",
        test_dir=dataset_path / "test",
        train_transform=transform,
        test_transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters()
    )

    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    test_loss, test_accuracy = trainer.test_step(
        test_loader
    )

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()