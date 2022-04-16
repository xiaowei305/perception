"""Encode target locations and labels."""
import math
import numpy as np
import torch


def nms(boxes, scores, threshold=0.3):
    """Non-maximum suppression algorithm.

    Args:
        boxes: (numpy.ndarray) bounding boxes, sized [N,4].
        scores: (numpy.ndarray) bounding box scores, sized [N,].

    Return:
        boxes: (numpy.ndarray) filtered bounding boxes, sized [N,4].
        scores: (numpy.ndarray) filtered bounding box scores, sized [N,]
    """
    box_and_scores = sorted(map(list, zip(boxes, scores)),
                            key=lambda x: x[1], reverse=True)
    for i, (box, score) in enumerate(box_and_scores):
        for j in range(i+1, len(box_and_scores)):
            if 0 < box_and_scores[j][1] < score and iou(box, box_and_scores[j][0]) > threshold:
                box_and_scores[j][1] = 0
    boxes = torch.stack([x[0] for x in box_and_scores if x[1] > 0])
    scores = torch.stack([x[1] for x in box_and_scores if x[1] > 0])
    return boxes, scores


def iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is
        [x1,y1,x2,y2].

    Args:
        box1: (tensor) bounding boxes, sized [N,4].
        box2: (tensor) bounding boxes, sized [N,4].

    Return:
        (tensor) iou, sized [N,].
    """
    lt = torch.max(box1[..., :2], box2[..., :2])
    rb = torch.min(box1[..., 2:], box2[..., 2:])

    wh = rb - lt  # [N,2]
    wh = torch.clamp(wh, 0)  # clip at 0
    inter = wh[..., 0] * wh[..., 1]  # [N,M]

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    return inter / (area1 + area2 - inter)


def generate_anchors(fm_sizes,
                     input_size=None,
                     anchor_sizes=None,
                     aspect_ratios=None,
                     ):
    """ Generates anchors for encoding.

    Args:
        fm_sizes: feature map sizes, e.g.
            [[44, 120], [22, 60], [11, 30], [6, 15], [3, 8], [2, 4], [1, 2]]
        input_size: image size of network input, for computing aspect ratio.
        anchor_sizes: normalized anchor sizes
        aspect_ratios: anchor aspect ratios for each anchor point.

    Returns:
        anchors: in xywh format and for each feature map layer.
    """
    input_aspect_ratio = float(input_size[1]) / float(input_size[0])

    steps = [(1.0 / h, 1.0 / w) for h, w in fm_sizes]

    anchors = []
    for i, (fh, fw) in enumerate(fm_sizes):
        min_size, max_size = anchor_sizes[i:i + 2]
        w, h = np.meshgrid(np.arange(fw), np.arange(fh))
        cx = (w + 0.5) * steps[i][1]
        cy = (h + 0.5) * steps[i][0]

        box = []

        for ar in aspect_ratios[i]:
            box_short = min_size / math.sqrt(ar)
            box_long = min_size * math.sqrt(ar)
            box.append([box_long, box_short])

        box.append([math.sqrt(min_size * max_size)] * 2)  # middle box 1:1
        anchor_num = len(aspect_ratios[i]) + 1

        wh = np.tile(box, (fh, fw, 1, 1))
        cxcy = np.tile((cx, cy),
                       (anchor_num, 1, 1, 1)).transpose(2, 3, 0, 1)
        box = np.concatenate((cxcy, wh), axis=3)
        box = box.reshape(-1, 4)
        box[:, 2] /= input_aspect_ratio
        box = box.reshape(-1, 4).tolist()
        anchors.append(box)
    return anchors


class DataEncoder(torch.nn.Module):

    def __init__(self,
                 input_size,
                 fm_sizes,
                 anchor_sizes,
                 aspect_ratios,
                 ):
        """Compute default box sizes with scale and aspect transform."""
        super().__init__()
        self.variances = [0.1, 0.2]
        anchors = generate_anchors(fm_sizes, input_size,
                                   anchor_sizes=anchor_sizes,
                                   aspect_ratios=aspect_ratios)
        boxes = [x for anchor in anchors for x in anchor]
        anchors = torch.tensor(boxes)
        self.register_buffer("default_boxes", anchors, persistent=False)

    @staticmethod
    def cross_iou(box1, box2):
        """Compute the intersection over union of two set of boxes, each box is
            [x1,y1,x2,y2].

        Args:
            box1: (tensor) bounding boxes, sized [B,N,4].
            box2: (tensor) bounding boxes, sized [M,4].

        Return:
            (tensor) iou, sized [N,M].
        """
        B = box1.size(0)
        N = box1.size(1)
        M = box2.size(0)
        box2 = box2.unsqueeze(0).expand(B, M, 4)

        lt = torch.max(  # xx1, yy1
            box1[:, :, :2].unsqueeze(2).expand(B, N, M, 2),
            box2[:, :, :2].unsqueeze(1).expand(B, N, M, 2))

        rb = torch.min(  # xx2, yy2
            box1[:, :, 2:].unsqueeze(2).expand(B, N, M, 2),
            box2[:, :, 2:].unsqueeze(1).expand(B, N, M, 2))

        wh = rb - lt  # [B,N,M,2]
        wh = torch.clamp(wh, min=0)  # clip at 0
        inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [B,N,M]

        area1 = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] -
                                                   box1[:, :, 1])
        area2 = (box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] -
                                                   box2[:, :, 1])
        area1 = area1.unsqueeze(2).expand_as(
            inter)  # [B,N,] -> [B,N,1] -> [B,N,M]
        area2 = area2.unsqueeze(1).expand_as(
            inter)  # [B,M,] -> [B,1,M] -> [B,N,M]

        return inter / (area1 + area2 - inter)  # [B, N, M]

    def encode(self, boxes, classes, nums, threshold=0.5):
        """ Transforms target bounding boxes and class labels to SSD boxes and
            classes.

        Match each object box to all the default boxes, pick the ones with
            the Jaccard-Index > 0.5: Jaccard(A,B) = AB / (A+B-AB)

        Args:
            boxes: (tensor) object bounding boxes (xmin,ymin,xmax,ymax) of an
                image, sized [#obj, 4].
            classes: (tensor) object class labels of a image, sized [#obj,].
            threshold: (float) Jaccard index threshold

        Returns:
            boxes: (tensor) bounding boxes, sized [#obj, #anchor, 4].
            classes: (tensor) class labels, sized [#anchor,]
        """
        default_boxes = self.default_boxes.type_as(boxes)
        num_default_boxes = default_boxes.size(0)
        B = boxes.size(0)
        if nums.sum() == 0:
            return (torch.zeros(B, num_default_boxes, 4),
                    torch.zeros(B, num_default_boxes).long())

        max_num = nums.max()
        boxes = boxes[:, :max_num]
        classes = classes[:, :max_num]

        iou = self.cross_iou(  # [B, #obj, #anchor]
            boxes,
            torch.cat([
                default_boxes[:, :2] - default_boxes[:, 2:] / 2,
                default_boxes[:, :2] + default_boxes[:, 2:] / 2
            ], 1))

        _, best_box_idx = iou.max(2, keepdim=True)  # [B, #obj, 1]
        best_box_idx = best_box_idx.expand_as(iou)  # [B, #obj, #anchor]
        best_iou_mask = iou.scatter(2, best_box_idx,
                                    2.0) > 1.0  # hold the best anchors
        iou[best_iou_mask] *= 5.0

        iou, max_idx = iou.max(1)  # [B,#anchor] # best box each anchor matched.
        max_idx_box = max_idx.unsqueeze(2).expand(-1, -1, 4)  # [B, #anchor, 4]
        boxes = torch.gather(boxes, 1, max_idx_box)  # [B, #anchor, 4]

        dxdy = (boxes[:, :, :2] +
                boxes[:, :, 2:]) / 2 - default_boxes[None, :, :2]  # [B, #anchor, 2]
        dxdy /= self.variances[0] * default_boxes[None, :, 2:]
        bwbh = boxes[:, :, 2:] - boxes[:, :, :2]
        dwdh = bwbh / default_boxes[None, :, 2:]  # [B, #anchor, 2]
        dwdh = torch.log(dwdh) / self.variances[1]
        loc = torch.cat([dxdy, dwdh], 2)  # [B,#anchor,4]

        conf = 1 + torch.gather(classes, 1,
                                max_idx)  # [B,#anchor], 0 is background
        mask = iou < threshold
        conf[mask] = 0  # background

        return loc, conf

    def decode_boxes(self, loc):
        """Transform [offset_x, offset_y, scale_w, scale_h] to [x, y, w, h]

        Args:
          loc: (tensor) encoded boxes, sized [#boxes,4].

        Returns:
          boxes: (tensor) decoded boxes , sized [#obj, 4].
        """

        default_boxes = self.default_boxes
        wh = torch.exp(loc[..., 2:] * self.variances[1]) * default_boxes[...,
                                                                         2:]
        cxcy = (loc[..., :2] * self.variances[0] * default_boxes[:, 2:] +
                default_boxes[:, :2])
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], -1)
        return boxes

    def forward(self, loc, conf):
        """ Transform predicted loc/conf back to real bbox locations and class
            labels.

        Args:
            loc: (tensor) predicted loc, sized [#anchor, 4].
            conf: (tensor) predicted conf, sized [#anchor, #class].

        Returns:
            boxes: (tensor) bbox locations, sized [#obj, 4].
            labels: (tensor) class labels, sized [#obj, 1].
        """
        conf = torch.sigmoid(conf)  # [N, #anchor, #class]
        boxes = self.decode_boxes(loc)  # [N, #anchor, 4]
        boxes = torch.clamp(boxes, 0.0, 1.0)
        boxes, scores, classes, keep, num = self.nms(boxes, conf)
        return boxes, scores, classes, keep, num
