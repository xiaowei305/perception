import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torchvision.models.mobilenet import mobilenet_v2

from encoder import DataEncoder


class MobileNetV2(nn.Module):
    """MobileNetV2 backbone"""
    def __init__(self):
        super().__init__()
        model = mobilenet_v2(pretrained=True)
        self.conv1to12 = model.features[:14]
        self.conv13to16 = model.features[14:]
        self.box_extractors = DownSample(input_channel=1280,
                                         output_channels=[512, 256, 256])

    def forward(self, x):
        """ Forwards"""
        output = []
        x = self.conv1to12(x)
        output.append(x)
        x = self.conv13to16(x)
        output.append(x)
        hs = self.box_extractors(x)
        output += hs
        return output


class MultiBoxLoss(nn.Module):

    def __init__(self, decoder):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss(reduction='mean')
        self.decoder = decoder

    @staticmethod
    def hard_negative_mining(conf_loss, pos):
        """Return negative indices that is 3x the number as postive indices.

        Args:
          conf_loss: (tensor) cross entroy loss between conf_preds and
              conf_targets, sized [N*#anchor,].
          pos: (tensor) positive(matched) box indices, sized [N,#anchor].

        Return:
          (tensor) negative indices, sized [N,#anchor].
        """
        batch_size, num_boxes, _ = pos.size()

        conf_loss[pos] = 0
        conf_loss = conf_loss.view(batch_size, -1)  # [N,#anchor * 21]

        _, idx = conf_loss.sort(1, descending=True)  # sort by neg conf_loss
        _, rank = idx.sort(1)  # [N, #anchor * 21]

        num_pos = pos.view(batch_size,
                           -1).long().sum(1)  # [N,#anchor * 21] -> [N,]
        num_neg = torch.clamp(3 * num_pos,
                              max=num_boxes - 1)[:, None]  # [N,] -> [N, 1]

        neg = rank < num_neg.expand_as(rank)  # [N, #anchor * 21]
        neg = neg.reshape_as(pos)  # [N, #anchor, 21]
        return neg

    def _iou(self, loc_preds, loc_targets, conf_targets):
        pos = conf_targets > 0
        box1 = self.decoder.decode_boxes(loc_preds)
        box2 = self.decoder.decode_boxes(loc_targets)
        box1 = box1[pos]
        box2 = box2[pos]
        return self.decoder.iou(box1, box2)

    @staticmethod
    def _statistic(conf_preds, conf_targets):
        pos = conf_targets > 0  # [N,#anchor], pos means the box matched.
        num_matched_boxes = pos.data.long().sum()
        if num_matched_boxes == 0:
            return {'accuracy': -1, 'recall': -1, 'iou': -1}

        # compute accuracy and recall
        conf, idx = conf_preds.max(2)
        tp = (conf > 0) & (idx + 1 == conf_targets)
        all_preds = conf > 0
        accuracy = (tp.float().sum() /
                    (all_preds.float().sum() + 0.0001)).item()
        recall = (tp.float().sum() / num_matched_boxes).item()

        info = {}
        info['accuracy'] = accuracy
        info['recall'] = recall
        return info

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        """Compute loss between (loc_preds, loc_targets) and (conf_preds,
        conf_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [B, #anchor, 4].
          loc_targets: (tensor) encoded target locations,
              sized [batch_size, #anchor, 4].
          conf_preds: (tensor) predicted class confidences,
              sized [batch_size, #anchor, num_classes].
          conf_targets: (tensor) encoded target classes,
              sized [batch_size, #anchor].

        Returns:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) +
                          CrossEntropyLoss(conf_preds, conf_targets).
        """
        batch_size, _, num_classes = conf_preds.size()

        pos = conf_targets > 0  # [N,#anchor], pos means the box matched.

        num_matched_boxes = pos.data.long().sum()
        info = {}
        if num_matched_boxes == 0:
            loss = {}
            loss['conf'] = (conf_preds.sum() * 0).view(-1)
            loss['loc'] = (loc_preds.sum() * 0).view(-1)
            return loss, info

        with torch.no_grad():
            info = self._statistic(conf_preds, conf_targets)
        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchor,4]
        pos_loc_preds = loc_preds[pos_mask].view(-1, 4)  # [#pos,4]
        pos_loc_targets = loc_targets[pos_mask].view(-1, 4)  # [#pos,4]
        iou = self._iou(loc_preds, loc_targets, conf_targets)
        with torch.no_grad():
            info['iou'] = iou.mean().item()
        loc_loss = F.smooth_l1_loss(pos_loc_preds,
                                    pos_loc_targets,
                                    reduction='mean')

        ################################################################
        # conf_loss = CrossEntropyLoss(pos_conf_preds, pos_conf_targets)
        #           + CrossEntropyLoss(neg_conf_preds, neg_conf_targets)
        ################################################################
        ones = torch.eye(num_classes + 1)
        conf_target_one_hot = ones.index_select(0, conf_targets.view(-1))
        conf_target_one_hot = conf_target_one_hot.view(batch_size, -1,
                                                       num_classes + 1)
        conf_target_one_hot = conf_target_one_hot[:, :, 1:]  # skip background type
        pos_mask = conf_target_one_hot > 0

        conf_loss = F.binary_cross_entropy_with_logits(
            conf_preds, conf_target_one_hot, reduction='none')  # [N,#anchor,21]
        neg_mask = self.hard_negative_mining(conf_loss, pos_mask)  # [N,#anchor]

        mask = pos_mask | neg_mask
        preds = conf_preds[mask]  # [#pos+#neg,]
        targets = conf_target_one_hot[mask]  # [#pos+#neg,]
        conf_loss = self.bce_loss(preds, targets)

        # conf_loss /= num_matched_boxes

        return {'loc': loc_loss.view(-1), 'conf': conf_loss.view(-1)}, info


class MultiBoxHead(nn.Module):
    """Simple box head with 1 layer convolution for cls and box"""
    def __init__(self, num_classes, in_channels, num_anchors):
        super().__init__()

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for c, a in zip(in_channels, num_anchors):
            self.loc_layers.append(
                nn.Conv2d(c, a * 4, kernel_size=3, padding=1))
            self.conf_layers.append(
                nn.Conv2d(c, a * num_classes, kernel_size=3, padding=1))
            self.num_classes = num_classes

        for layer in [self.loc_layers, self.conf_layers]:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, xs):
        """
        Args:
          xs: (list) of tensor containing intermediate layer outputs.

        Returns:
          loc_preds: (tensor) predicted locations, sized [N,#anchor,4].
          conf_preds: (tensor) predicted class confidences, sized [N,#anchor,21].
        """
        y_locs = []
        y_confs = []
        for i, x in enumerate(xs):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()  # N, H, W, C
            y_loc = y_loc.view(N, -1, 4)  # N, #anchor, 4
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)  # N, #anchor, #class
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)
        return loc_preds, conf_preds


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        """ Forwards"""
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class DownSample(nn.Module):
    """ 2X down sample by 2 consecutive convolutions, following SSD, e.g
        Conv(256, 128, kernel_size=1, padding=0, stride=1)
        Conv(128, 256, kernel_size=3, padding=1, stride=2)
        The first 1x1 convolution will use num_output_channels / 2 channels,
        and the second 3x3 convolution will use stride=2 by default.
        You can also set other strides and padding parameters.
        It's used for adding more feature map layers given input feature maps:
        [80, 80], [40, 40], [20, 20]
        we add 2 down sample layers, then the output will be
        [80, 80], [40, 40], [20, 20], [10, 10], [5, 5]
    """
    def __init__(self,
                 input_channel,
                 output_channels,
                 strides=None,
                 paddings=None,
                 quantize=False):
        super().__init__()
        self.box_extractor = nn.ModuleList()
        if strides is None:
            strides = [2 for _ in output_channels]
        if paddings is None:
            paddings = [1 for _ in output_channels]
        for channel, stride, padding in zip(output_channels, strides,
                                            paddings):
            self.append_down_sample_conv(input_channel, channel, stride,
                                         padding, quantize)
            input_channel = channel

    def append_down_sample_conv(self, inc, outc, stride, padding, quantize):
        """ Adds box extractor"""
        outc1 = outc // 2
        conv1 = BasicConv2d(inc, outc1, kernel_size=1)
        conv2 = BasicConv2d(outc1,
                          outc,
                          kernel_size=3,
                          padding=padding,
                          stride=stride)
        extractor = nn.Sequential(conv1, conv2)
        self.box_extractor.append(extractor)

        for param in extractor.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        """ Forwards"""
        output = []
        for layer in self.box_extractor:
            x = layer(x).contiguous()
            output.append(x)

        return output


class SSD(nn.Module):
    """SSD model"""

    def __init__(self,
                 num_classes=8,
                 input_size=(185, 612),
                 anchor_sizes=(0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0),
                 aspect_ratios=(1.0, 2.0, 0.5, 3.0, 0.3333),
                 max_output_per_class=100,
                 max_output=100,
                 nms_iou_threshold=0.4,
                 nms_score_threshold=0.2,
                 nms_together=True,
                 ):
        super().__init__()
        backbone = MobileNetV2()

        # get feature map size of box layers
        print("input_size: {}".format(input_size))
        random_input = torch.rand(2, 3, *input_size)
        backbone.eval()
        with torch.no_grad():
            features = backbone(random_input)
        fm_sizes = [list(x.shape[2:]) for x in features]
        print("fm_size = {}".format(fm_sizes))
        box_channels = [x.shape[1] for x in features]
        print("box_channels = {}".format(box_channels))

        # multibox layer
        num_anchors = [len(aspect_ratios) + 1 for x in range(len(fm_sizes))]
        print("num_anchors = {}".format(num_anchors))
        self.multibox = MultiBoxHead(num_classes=num_classes, in_channels=box_channels,
                                     num_anchors=num_anchors)
        aspect_ratios = [aspect_ratios] * len(fm_sizes)
        self.decoder = DataEncoder(input_size, fm_sizes, anchor_sizes,
                                   aspect_ratios)
        self.input_size = input_size

        self.multibox_loss = MultiBoxLoss(self.decoder)
        self.backbone = backbone

    def forward(self, x, targets=None):
        """ Forward"""
        hs = self.backbone(x)
        loc_preds, conf_preds = self.multibox(hs)
        outputs = {}
        if self.training:  # training
            with torch.no_grad():
                threshold = 0.5

                loc_targets, conf_targets = self.decoder.encode(
                        targets['boxes'], targets['classes'],
                        targets['num_boxes'], threshold)

            box_loss, box_info = self.multibox_loss(loc_preds, loc_targets,
                                                    conf_preds, conf_targets,
                                                    )
            outputs.update(box_info)
            outputs.update(box_loss)

            return outputs

        with torch.no_grad():
            boxes, scores, classes, selected, nums = self.decoder(
                loc_preds, conf_preds)

        outputs.update({
            "detection_boxes": boxes,
            "detection_scores": scores,
            "detection_classes": classes,
            "num_detections": nums
        })

        return outputs
