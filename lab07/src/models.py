import torch
from torch import nn

class ConvNet(nn.Module):
    """CNN with configurable channel widths and classifier width."""
    def __init__(self, channels=(32, 64, 96), hidden=512, pool_size=4):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.adaptive = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * pool_size * pool_size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive(x)
        return self.classifier(x)

def make_teacher():
    # Approximately 0.9M trainable parameters; verify exact count at runtime.
    return ConvNet(channels=(32, 64, 96), hidden=512, pool_size=4)

def make_student():
    # Compact student, approximately 100k parameters.
    return ConvNet(channels=(24, 32, 64), hidden=256, pool_size=2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
