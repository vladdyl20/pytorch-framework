import csv
import json
from pathlib import Path


def save_history_json(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def save_history_csv(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = len(history["train_loss"])

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
            ]
        )

        for epoch in range(epochs):
            writer.writerow(
                [
                    epoch + 1,
                    history["train_loss"][epoch],
                    history["train_accuracy"][epoch],
                    history["test_loss"][epoch],
                    history["test_accuracy"][epoch],
                ]
            )