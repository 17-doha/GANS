import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        # Dense layer
        self.fc1 = nn.Linear(noise_dim, 128 * 6 * 6)
        self.lrelu = nn.LeakyReLU(0.2)
        
        # Transposed Convolutions (Deconvolutions)
        # padding=0 is the equivalent of Keras's padding='valid'
        self.convt1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=0)
        self.convt2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=0)
        
        # Final Convolution
        self.conv_final = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        # Reshape to (Batch Size, Channels, Height, Width)
        x = x.view(-1, 128, 6, 6) 
        
        x = self.lrelu(self.convt1(x))  # Outputs: (128, 14, 14)
        x = self.lrelu(self.convt2(x))  # Outputs: (128, 30, 30)
        x = self.sigmoid(self.conv_final(x)) # Outputs: (1, 28, 28)
        return x

def building_generator(noise_dim):
    return Generator(noise_dim)