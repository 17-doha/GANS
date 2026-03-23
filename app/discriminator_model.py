import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # PyTorch expects (Channels, Height, Width) -> (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0)
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.4)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        
        self.flatten = nn.Flatten()
        # After two valid stride-2 convs on 28x28, the spatial size is 6x6
        self.fc1 = nn.Linear(64 * 6 * 6, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.lrelu(self.conv1(x)))
        x = self.dropout(self.lrelu(self.conv2(x)))
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        return x

def building_discriminator(learning_rate=None):
    return Discriminator()