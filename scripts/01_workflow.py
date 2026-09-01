import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            in_features=1,
            out_features=1,
        )

    def forward(self, x):
        return self.linear(x)


def main():
    torch.manual_seed(42)

    weight = 0.7
    bias = 0.3

    x = torch.arange(
        0,
        1,
        0.02,
    ).unsqueeze(dim=1)

    y = weight * x + bias

    train_split = int(0.8 * len(x))

    x_train = x[:train_split]
    y_train = y[:train_split]

    x_test = x[train_split:]
    y_test = y[train_split:]

    model = LinearRegressionModel()

    loss_fn = nn.L1Loss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    epochs = 200

    for epoch in range(epochs):
        model.train()

        predictions = model(x_train)

        loss = loss_fn(
            predictions,
            y_train,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 20 == 0:
            model.eval()

            with torch.inference_mode():
                test_predictions = model(x_test)

                test_loss = loss_fn(
                    test_predictions,
                    y_test,
                )

            print(
                f"Epoch: {epoch} | "
                f"Train loss: {loss:.4f} | "
                f"Test loss: {test_loss:.4f}"
            )

    model.eval()

    with torch.inference_mode():
        final_predictions = model(x_test)

    print()
    print("Learned parameters:")
    print(model.state_dict())

    print()
    print("Predictions:")
    print(final_predictions[:5])


if __name__ == "__main__":
    main()