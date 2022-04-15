import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, Resize, ToTensor

from net import Net

model = Net()

transform = Compose([Resize(size=[32, 32]), ToTensor()])
dataset = MNIST(root='./datasets/', download=True, train=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# call net.cuda(), inputs.cuda(), labels.cuda() to use GPU
# net.cuda()
for epoch in range(20):
    for i, (inputs, labels) in enumerate(dataloader):
        # inputs = inputs.cuda()
        # labels = labels.cuda()
        optimizer.zero_grad()  # 清空优化器状态
        outputs = model(inputs)  # 运行网络，得到输出结果
        loss = criterion(outputs, labels)  # 计算loss
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        prediction = torch.argmax(outputs, dim=1)  # 计算预测结果
        true_positive = torch.sum(prediction == labels)  # 计算样本预测准确的数量
        accuracy = true_positive / len(labels)  # 计算当前batch的精度
        print("epoch {} step {}: loss = {:.3f}, accuracy = {:.3f}".format(
            epoch, i, loss.item(), accuracy.item()))

torch.save(model.state_dict(), 'model.pth')
