import torch
from tqdm import tqdm

from pytorch_framework.engine.base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        device,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_step(self, dataloader):
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

        loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return loss, accuracy

    def test_step(self, dataloader):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.inference_mode():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

                total_loss += loss.item() * images.size(0)

                predictions = logits.argmax(dim=1)

                total_correct += (
                    predictions == labels
                ).sum().item()

                total_samples += labels.size(0)

        loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return loss, accuracy

    def fit(
        self,
        train_loader,
        test_loader,
        epochs,
    ):
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        for epoch in tqdm(range(epochs)):
            train_loss, train_accuracy = self.train_step(
                train_loader
            )

            test_loss, test_accuracy = self.test_step(
                test_loader
            )

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"train acc: {train_accuracy:.4f} | "
                f"test loss: {test_loss:.4f} | "
                f"test acc: {test_accuracy:.4f}"
            )

        return history