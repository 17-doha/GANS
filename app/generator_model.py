import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        # 1. Project noise to a 7x7 spatial size
        self.fc1 = nn.Linear(noise_dim, 128 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(128 * 7 * 7)
        self.relu = nn.ReLU(True)

        # 2. Upsample to 14x14
        self.convt1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # 3. Upsample to 28x28
        self.convt2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )

        # Output layer uses Sigmoid to keep pixels between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 128, 7, 7)

        x = self.relu(self.bn2(self.convt1(x)))
        x = self.sigmoid(self.convt2(x))
        return x


def building_generator(noise_dim):
    return Generator(noise_dim)
