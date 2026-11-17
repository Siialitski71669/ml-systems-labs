import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    model = CNN()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"\nTrainable parameters: {parameters:,}")

    assert parameters == 20490
    print("CNN architecture verified successfully.")
