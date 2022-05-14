import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lanenet import LaneNet
from data import TuSimple


def collate_fn(batch_data):
    images, targets = list(zip(*batch_data))
    merged_targets = dict(
        (k, [x[k] for x in targets]) for k in targets[0].keys())

    return torch.stack(images), merged_targets


def main():
    model = LaneNet()
    dataset = TuSimple(root='D:\\datasets\\tusimple\\train_set')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-5, momentum=0.9)

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


if __name__ == "__main__":
    main()
