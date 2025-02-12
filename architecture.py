import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class LeNet(nn.Module):
    """
    LeNet Convolutional Neural Network.

    Architecture based on:
    Yann LeCun et al., "Gradient-Based Learning Applied to Document Recognition"
    (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


pre_processing = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(28, 28)),
        transforms.Normalize((0.5,), (0.5)),
    ]
)
