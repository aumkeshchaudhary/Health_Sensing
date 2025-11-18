import torch
import torch.nn as nn

class Simple1DCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.fc(h)
