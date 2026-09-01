from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    path,
    model_name,
    class_names,
    model_config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "model_config": model_config,
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    return checkpoint