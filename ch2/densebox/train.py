import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets.kitti import Kitti
from torchvision.transforms import ToTensor

from densebox import DenseBox


def collate_fn(batch_data):
    class_names = ['Car', 'Truck', 'Cyclist', 'Tram', 'Person_sitting',
                   'Misc', 'Van', 'Pedestrian']
    images, targets = list(zip(*batch_data))
    num_boxes = [len(x) for x in targets]
    max_boxes = max(num_boxes)

    all_boxes = []
    all_classes = []
    for img, target in batch_data:
        c, h, w = img.shape
        boxes = np.array([x["bbox"] for x in target if x['type'] != "DontCare"])
        boxes = np.pad(boxes, ((0, max_boxes - len(boxes)), (0, 0)))
        boxes /= np.array([w, h, w, h])
        all_boxes.append(boxes)
        classes = np.array([class_names.index(x['type']) for x in target if x['type'] != "DontCare"])
        classes = np.pad(classes, (0, max_boxes - len(classes)))
        all_classes.append(classes)
    all_boxes = torch.from_numpy(np.stack(all_boxes, axis=0))
    all_classes = torch.from_numpy(np.stack(all_classes, axis=0))

    targets = {
        "boxes": all_boxes,
        "classes": all_classes,
        "num_boxes": np.array(num_boxes)
    }
    images = [F.interpolate(x[None], (185, 612), mode="bilinear") for x in images]
    return torch.cat(images), targets


def main():
    model = DenseBox()
    dataset = Kitti(root='C:\\Users\\yangyang\\datasets\\KITTI',
                    download=False, train=True,
                    transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)

    # call net.cuda(), inputs.cuda(), labels.cuda() to use GPU
    # net.cuda()
    for epoch in range(20):
        for i, (inputs, labels) in enumerate(dataloader):
            # inputs = inputs.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()  # 清空优化器状态
            losses = model(inputs, labels)  # 运行网络，得到loss
            total_loss = 0
            loss_str = ""
            for name, loss in losses.items():
                total_loss += loss
                loss_str += f" {name}: {float(loss):.3f}"
            total_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print("epoch {} step {}: loss = {:.3f} {}".format(
                epoch, i, total_loss.item(), loss_str))

    torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()
