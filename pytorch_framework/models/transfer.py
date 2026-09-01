from torch import nn
from torchvision import models


class EfficientNetB0Transfer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_features: bool = True,
    ):
        super().__init__()

        self.weights = models.EfficientNet_B0_Weights.DEFAULT

        self.model = models.efficientnet_b0(
            weights=self.weights
        )

        if freeze_features:
            for parameter in self.model.features.parameters():
                parameter.requires_grad = False

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=in_features,
                out_features=num_classes,
            ),
        )

    def forward(self, x):
        return self.model(x)

    def get_transforms(self):
        return self.weights.transforms()