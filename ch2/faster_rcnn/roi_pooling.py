import torch
import  torch.nn.functional as F


def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):
    """ROI Pooling

    Args:
        input: feature map ,sized (N, C, H, W)
        rois: boxes, sized (#roi, 5) where 0 is the batch index, 1~4 is box
        size: roi pooling output size
        spatial_scale: if the roi is normalized, the spatial_scale should
            be the feature map size.
    """
    output = []
    rois = rois.data.float()

    rois[:, 1:] = rois[:, 1:] * spatial_scale
    rois = rois.long()
    for roi in rois:
        batch_idx = roi[0]
        im = input[batch_idx, :, roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        output.append(F.adaptive_max_pool2d(im, size))

    output = torch.cat(output, 0)
    return output
