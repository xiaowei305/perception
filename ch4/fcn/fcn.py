from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models import resnet18


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class FCN(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        backbone = resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:6])

        self.classifier = FCNHead(128, num_classes)
        self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)

    def forward(self, x: Tensor, label: Tensor) -> Tensor:
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        loss = self.loss(x, label[:, 0].long())
        return loss
