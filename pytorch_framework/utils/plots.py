from pathlib import Path

import matplotlib.pyplot as plt


def save_training_plots(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train loss",
    )

    plt.plot(
        epochs,
        history["test_loss"],
        label="Test loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid()

    plt.savefig(
        output_dir / "loss.png",
        bbox_inches="tight",
    )

    plt.close()

    plt.figure()

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train accuracy",
    )

    plt.plot(
        epochs,
        history["test_accuracy"],
        label="Test accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.grid()

    plt.savefig(
        output_dir / "accuracy.png",
        bbox_inches="tight",
    )

    plt.close()