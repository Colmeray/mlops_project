from torch import nn
import torch
from torchvision import models

## FIXME: billeder skal preprocess til 224x224 før det virker
# VGG16 bare midlertidigt, :


class SimpleModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGG16Transfer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_features: bool = True,
        weights=models.VGG16_Weights.DEFAULT,
    ):
        super().__init__()
        self.backbone = models.vgg16(weights=weights)

        if freeze_features:
            for p in self.backbone.features.parameters():
                p.requires_grad = False

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = SimpleModel(num_classes=10)
    x = torch.rand(1, 3, 224, 224)
    print(f"Output shape of model: {model(x).shape}")
