from pathlib import Path

import torch
from torch import nn

from pytorch_framework.data.download import download_dataset
from pytorch_framework.data.loaders import create_dataloaders
from pytorch_framework.engine.classification import ClassificationTrainer
from pytorch_framework.models.transfer import EfficientNetB0Transfer
from pytorch_framework.utils.checkpoints import save_checkpoint
from pytorch_framework.utils.logging import save_history_csv, save_history_json
from pytorch_framework.utils.plots import save_training_plots
from pytorch_framework.utils.seed import set_seed


def main():
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset_path = download_dataset()

    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    model = EfficientNetB0Transfer(
        num_classes=3
    )

    transform = model.get_transforms()

    train_loader, test_loader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=transform,
        test_transform=transform,
        batch_size=16,
        num_workers=0,
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=0.001,
    )

    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Classes: {class_names}")

    history = trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=2,
    )

    run_dir = Path("runs") / "transfer_learning"

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
        path=Path("checkpoints") / "transfer_learning.pth",
        model_name="efficientnet_b0",
        class_names=class_names,
        model_config={
            "num_classes": len(class_names),
        },
    )


if __name__ == "__main__":
    main()