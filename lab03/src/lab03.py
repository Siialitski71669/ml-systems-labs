import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_projection = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Split each 28x28 image into 16 patches of 7x7 pixels.
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(x.size(0), 16, 49)

        x = self.patch_projection(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    for name, model in models.items():
        parameters = sum(p.numel() for p in model.parameters())
        print(f"{name}: {parameters:,} parameters")

    print("\nArchitecture file created successfully.")
