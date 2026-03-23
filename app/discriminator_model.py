import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # 1. Downsample 28x28 to 14x14 (No BatchNorm on first discriminator layer)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(0.3)
        
        # 2. Downsample 14x14 to 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        # 3. Flatten and output probability
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.lrelu(self.conv1(x)))
        x = self.dropout2(self.lrelu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        return x

def building_discriminator(learning_rate=None):
    return Discriminator()