import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torchvision.transforms import InterpolationMode

from deeplabv3 import DeepLabV3


def main():
    model = DeepLabV3(num_classes=35)
    transforms = Compose([
        Resize(size=(256, 512)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transforms = Compose([
        Resize(size=(256, 512), interpolation=InterpolationMode.NEAREST),
        ToTensor(),
    ])
    dataset = Cityscapes(root='D:\\datasets\\cityscapes',
                         split="train",
                         target_type="semantic",
                         transform=transforms,
                         target_transform=target_transforms
                         )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)

    # call net.cuda(), inputs.cuda(), labels.cuda() to use GPU
    # net.cuda()
    for epoch in range(20):
        for i, (inputs, labels) in enumerate(dataloader):
            # inputs = inputs.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()  # 清空优化器状态
            loss = model(inputs, labels)  # 运行网络，得到loss
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print("epoch {} step {}: loss = {:.3f}".format(
                epoch, i, loss.item()))

    torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()
