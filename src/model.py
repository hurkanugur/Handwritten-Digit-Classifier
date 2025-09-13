import torch
from torch import nn
import config

class MNISTClassificationModel(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),               # Converts input tensor to 1D tensor (e.g., 1x28x28 -> 784)

            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 10)           # Output layer for 10 classes (MNIST)
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
        self.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
        self.to(self.device)
        print(f"• Model loaded from {config.MODEL_PATH}")
