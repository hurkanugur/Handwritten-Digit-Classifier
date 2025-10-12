import torch
from torch import nn
from src import config

class MNISTClassificationModel(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.net = nn.Sequential(
            # Conv Block 1:
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            ),
            # Conv Block 2:
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            ),
            # Fully Connected:
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )
        )

        self.net.apply(self.init_weights)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def save(self):
        torch.save(self.state_dict(), config.MODEL_PATH)
        print(f"• Model saved to {config.MODEL_PATH}")

    def load(self):
        self.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True, map_location=self.device))
        self.to(self.device)
        print(f"• Model loaded from {config.MODEL_PATH}")
