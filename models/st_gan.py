import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Initial convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Dilated convolutional layers
        self.dilated_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dilated_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dilated_conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=16, dilation=16)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=0)
        self.conv10 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # Activation
        self.relu = nn.ReLU()

        # Average pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        # Spatio-temporal feature extraction
        d1 = self.relu(self.dilated_conv1(x6))
        d2 = self.relu(self.dilated_conv2(d1))
        d3 = self.relu(self.dilated_conv3(d2))
        d4 = self.relu(self.dilated_conv4(d3))
        x7 = self.relu(self.conv7(d4))
        x8 = self.relu(self.conv8(x7))

        x9 = self.relu(self.deconv1(x8))
        x10 = self.relu(self.conv9(x9 + x3))
        x11 = self.relu(self.deconv2(x10))
        x12 = self.relu(self.conv10(x11 + x1))
        return x12


# Example usage
generator = Generator()
incomplete_data = torch.tensor(np.random.rand(10, 1, 64, 64),
                               dtype=torch.float32)  # Example tensor with incomplete data
imputed_data = generator(incomplete_data)
print(imputed_data.shape)  # Should match the input shape (10, 1, 64, 64)
