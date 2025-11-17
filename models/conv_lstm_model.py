import torch
import torch.nn as nn

class ConvLSTMModel(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, num_layers=1, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, 7, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)

        self.lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.pool(torch.relu(self.conv1(x)))
        h = torch.relu(self.conv2(h))
        h = h.permute(0, 2, 1)

        out, (hn, _) = self.lstm(h)
        return self.fc(out[:, -1, :])
