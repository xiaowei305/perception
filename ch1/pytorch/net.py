import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(3 * 3 * 64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=2))
        x = F.relu(F.avg_pool2d(self.conv2(x), 3, stride=2))
        x = F.relu(F.avg_pool2d(self.conv3(x), 3, stride=2))
        x = x.view(-1, 3 * 3 * 64)
        x = self.fc1(x)
        return x
