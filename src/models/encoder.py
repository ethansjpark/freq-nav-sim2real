import torch.nn as nn
import torch

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Replace with ResNet18 or ViT
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * 54 * 54, 512)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)
