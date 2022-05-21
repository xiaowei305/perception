import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pointnet2 import PointNet2SemSeg
from dataset import SemKITTI


def main():
    model = PointNet2SemSeg()
    dataset = SemKITTI("D:\\datasets\\KITTI\\data_odometry_velodyne\\dataset", npoints=50000)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)

    # call net.cuda(), inputs.cuda(), labels.cuda() to use GPU
    # net.cuda()
    for epoch in range(20):
        for i, (inputs, labels) in enumerate(dataloader):
            # inputs = inputs.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()
            loss = model(inputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print("epoch {} step {}: loss = {:.3f}".format(
                epoch, i, loss.item()))

    torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()
