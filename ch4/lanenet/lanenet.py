from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from enet import ENet


class LaneNet(nn.Module):
    def __init__(self, input_size=(280, 640), embedding_channels=8):
        super().__init__()
        backbone = ENet(num_classes=1 + embedding_channels)

        random_input = torch.rand(2, 3, *input_size)
        features = backbone(random_input)
        print(features.shape)
        self.embedding_channels = embedding_channels
        self.fm_h = int(features.shape[2])
        self.fm_w = int(features.shape[3])
        self.backbone = backbone
        self.ce_loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def discriminative_loss(preds,
                            targets,
                            mask,
                            delta_v=0.2,
                            delta_d=1.2):
        N, C, H, W = preds.shape
        with torch.no_grad():
            group_target = targets.long()  # [N, H, W]
            group_target[mask] = group_target[mask] - 1  # [N, H, W]
            group_num = group_target.max() + 1  # group id starts from zero
            group_target = F.one_hot(group_target).view(N, H * W, group_num)
            mask = mask.logical_not().unsqueeze(3).view(N, H * W, 1).expand_as(group_target)  # [N, H*W, #group]
            group_target[mask] = 0  # [N, H*W, #group]

            group_mask = group_target.unsqueeze(3).expand(N, H * W, group_num, C)  # [N, H*W, #group, 8]  one hot
            point_num = group_mask.sum(1, keepdim=True)  # [N, 1, #group, 8]
            has_point = (point_num > 0).float()[:, 0, :, 0]  # [N, #group]
            n_groups = has_point.sum(1).unsqueeze(1)  # [N, 1]
            has_point2 = has_point.unsqueeze(2)  # [N, #group, 1]

        group_pred = preds.permute(0, 2, 3, 1).view(N, H * W,
                                                    -1)  # [N, H*W, 8]
        group_pred = group_pred.unsqueeze(2)  # [N, H*W, 1, 8]

        group_center = (group_pred * group_mask).sum(1, keepdim=True)  # [N, 1, #group, 8]
        group_center = group_center / (point_num + 0.00001)
        loss_var = torch.norm(group_pred - group_center, dim=3)  # [N, H*W, #group]
        loss_var = torch.clamp(loss_var - delta_v, min=0.0) ** 2 * group_target  # [N, H*W, #group]
        loss_var = loss_var.sum((1, 2)) / n_groups
        loss_var = loss_var.mean()

        group_center1 = group_center.squeeze(1).unsqueeze(2)  # [N, #group, 1, 8]
        distance = torch.norm(group_center - group_center1, dim=3)  # [N, #group, #group]
        loss_dist = torch.clamp(delta_d - distance, min=0.0) ** 2 * has_point2
        loss_dist = loss_dist.sum((1, 2)) / torch.clamp(n_groups * (n_groups - 1), min=1.0)  # [N, 1]
        loss_dist = loss_dist.mean()

        mask = has_point > 0
        loss_reg = group_center.sum((1, 3))[mask] ** 2 * 0.001
        return {
            "dist": loss_dist,
            "var": loss_var * 0.01,
            "reg": loss_reg.sum()
        }

    def compute_loss(self, preds, targets):
        N, _, H, W = preds.size()
        mask = targets[:, 0].sum(2).sum(1) != 0
        if mask.sum() == 0:
            return {
                "loc": preds.abs().sum() * 0.0,
                "conf": preds.abs().sum() * 0.0
            }
        preds = preds[mask]
        targets = targets[mask]

        points_pred = preds[:, 0]
        points_target = targets[:, 0]
        points_loss = self.ce_loss(points_pred, points_target)

        mask = points_target > 0  # [N, H, W]
        loss = {}
        loss['conf'] = points_loss.view(-1)
        loss_embedding = self.discriminative_loss(preds[:, 1:], targets[:, 1], mask)
        loss.update(loss_embedding)
        return loss

    @staticmethod
    def encode(targets, fm_h, fm_w):
        """
        Args:
            fm_h: height of feature map
            fm_w: width of feature map
            targets: list(list([xs, ys])), where xs and ys are normalized to 0 ~ 1.

        Returns:
            target: Torch.Tensor
        """
        y_samples = np.arange(0, fm_h)
        target = np.zeros(
            (len(targets), 3, fm_h, fm_w))  # 3 = points, offset, group

        for i, lane_group in enumerate(targets):
            lane_id = 1
            for lane in lane_group:
                if isinstance(lane, torch.Tensor):
                    lane = lane.detach().cpu().numpy()
                xs = lane[:, 0]
                ys = lane[:, 1]
                # in some corner case, lane_y may repeat, which will lead
                # some runtime error
                ys, index = np.unique(ys, return_index=True)
                xs = np.array(xs)[index]
                if len(xs) < 2:
                    continue
                interp = interp1d(ys, xs, kind='linear')

                min_y = min(ys)
                max_y = max(ys)
                point_min_y = min_y * fm_h
                point_max_y = max_y * fm_h
                points_y = y_samples[(y_samples >= point_min_y) &
                                     (y_samples <= point_max_y)]
                points_x = interp(points_y / fm_h) * fm_w
                points_x_index = np.array(points_x).round().astype(np.int64)
                mask = (points_x_index < fm_w) & (points_x_index >= 0)
                if mask.sum() < 2:
                    continue
                points_x = points_x[mask]
                points_y = points_y[mask]
                points_x_index = points_x_index[mask]

                points_x_offset = points_x - points_x_index
                target[i, 0, points_y, points_x_index] = 1
                target[i, 1, points_y, points_x_index] = lane_id
                target[i, 2, points_y, points_x_index] = points_x_offset
                lane_id += 1

        target = torch.tensor(target).float()
        return target

    def forward(self, x, targets=None):
        preds = self.backbone(x)

        if self.training:
            with torch.no_grad():
                targets = self.encode(targets["lane"], self.fm_h, self.fm_w)
                targets = targets.to(preds.device)
            loss = self.compute_loss(preds, targets)
            return loss
