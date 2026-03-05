import torch.nn as nn
import torch.nn.functional as F
import torch


# simple convolutional neural network with (4, 3, 416, 416) input and 4 output numbers for the 4 facial landmarks (x, y) coordinates
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 101 * 101, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.conv1(x)  # Output shape: (4, 6, 412, 412)
        x = self.pool(F.relu(x))  # Output shape: (4, 6, 206, 206)
        x = self.conv2(x)  # Output shape: (4, 16, 202, 202)
        x = self.pool(
            F.relu(x)
        )  # Output shape: (4, 16, 101, 101) - This is the new shape after pooling

        # Make sure the view matches the new calculation
        x = x.view(-1, 16 * 101 * 101)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
