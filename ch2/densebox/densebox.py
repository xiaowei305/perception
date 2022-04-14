import torch
from torchvision.models.mobilenet import mobilenet_v2
import numpy as np


class DenseBox(torch.nn.Module):
    def __init__(self, num_classes=8, feature_height=6, feature_width=20):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=True)  # torchvision中的mobilenet_v2
        self.num_classes = num_classes
        h, w = torch.meshgrid(torch.arange(feature_height),
                              torch.arange(feature_width))  # 生成中心点
        centers = torch.stack((w, h, w, h), axis=2) + 0.5
        centers = centers.reshape(feature_height, feature_width, 4).permute(2, 0, 1)
        self.conv = torch.nn.Conv2d(1280, num_classes + 4, kernel_size=3, padding=1)
        self.register_buffer("centers", centers.float(), persistent=False)  # self.center=center
        self.ce_loss = torch.nn.CrossEntropyLoss()  # 分类loss
        self.l2_loss = torch.nn.MSELoss()  # box回归的loss

    def forward(self, images, targets):
        features = self.backbone.features(images)
        features = self.conv(features)
        n, _, h, w = features.shape  # [B, 8+4, 6, 20]
        cls_target = torch.zeros((n, self.num_classes, h, w), dtype=torch.float)
        reg_target = torch.zeros((n, 4, h, w), dtype=torch.float)
        ratio = 0.3
        for i in range(n):
            for box, cls in zip(targets["boxes"][i], targets["classes"][i]):
                box = box * np.array([w, h, w, h])
                bbox_center_x = (box[0] + box[2]) * 0.5
                bbox_center_y = (box[1] + box[3]) * 0.5

                bbox_w = box[2] - box[0]
                bbox_h = box[3] - box[1]

                # 计算正样本区域
                org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
                org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
                end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
                end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
                cls_target[i, :, org_y: end_y + 1, org_x: end_x + 1] = cls + 1.0
                reg_target[i, :, org_y: end_y + 1, org_x: end_x + 1] = (
                    self.centers[:, org_y: end_y + 1, org_x: end_x + 1]
                    - torch.tensor(box, dtype=torch.float)[:, None, None])

        cls_loss = self.ce_loss(features[:, :self.num_classes, :, :], cls_target)
        reg_loss = self.l2_loss(features[:, self.num_classes:, :, :], reg_target)
        return {"cls": cls_loss, "reg": reg_loss}
