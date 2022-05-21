import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pointnet import PointNetSeg
from dataset import SemKITTI


def main():
    model = PointNetSeg()
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
            losses = model(inputs, labels)
            total_loss = 0
            loss_str = ""
            for name, loss in losses.items():
                total_loss += loss
                loss_str += f" {name}: {float(loss):.3f}"
            total_loss.backward()
            optimizer.step()
            print("epoch {} step {}: loss = {:.3f} {}".format(
                epoch, i, total_loss.item(), loss_str))

    torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()
