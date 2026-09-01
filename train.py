import argparse
from pathlib import Path

import torch
from torch import nn

from pytorch_framework.data.download import download_dataset
from pytorch_framework.data.loaders import create_dataloaders
from pytorch_framework.data.transforms import (
    get_basic_transform,
    get_train_transform,
)
from pytorch_framework.engine.classification import ClassificationTrainer
from pytorch_framework.models.tiny_vgg import TinyVGG
from pytorch_framework.utils.checkpoints import save_checkpoint
from pytorch_framework.utils.logging import save_history_csv, save_history_json
from pytorch_framework.utils.plots import save_training_plots
from pytorch_framework.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--hidden-units",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="tinyvgg_baseline",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    dataset_path = download_dataset()

    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    train_transform = get_train_transform(
        image_size=args.image_size
    )

    test_transform = get_basic_transform(
        image_size=args.image_size
    )

    train_loader, test_loader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    model = TinyVGG(
        input_channels=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names),
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    history = trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
    )

    run_dir = Path("runs") / args.experiment

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_history_json(
        history,
        run_dir / "metrics.json",
    )

    save_history_csv(
        history,
        run_dir / "metrics.csv",
    )

    save_training_plots(
        history,
        run_dir,
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        path=Path("checkpoints") / f"{args.experiment}.pth",
        model_name="tinyvgg",
        class_names=class_names,
        model_config={
            "input_channels": 3,
            "hidden_units": args.hidden_units,
            "output_shape": len(class_names),
            "image_size": args.image_size,
        },
    )

    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()