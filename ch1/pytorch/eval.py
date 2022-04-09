#_*_coding:utf8_*_
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, Resize, ToTensor

from net import Net

model= Net()

transform = Compose([Resize(size=[32, 32]), ToTensor()])
dataset= MNIST(root='./datasets/', download=True, train=False, transform=transform)
dataloader= DataLoader(dataset, batch_size=1000, shuffle=False)
model.load_state_dict(torch.load('model.pth'))

true_positive = 0
for inputs, labels in dataloader:
    outputs = model(inputs)
    prediction = torch.argmax(outputs, dim=1)
    true_positive += torch.sum(prediction.eq(labels))
accuracy = true_positive / len(dataset)
print("accuracy = {:.3f}".format(accuracy))
