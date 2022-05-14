import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from cv2 import imread, imwrite


def CRFs(original_image_path, predicted_image_path, CRF_image_path):
    print("original_image_path: ", original_image_path)
    img = imread(original_image_path)

    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    anno_rgb = cv2.imread(predicted_image_path).astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))

    # 使用densecrf类
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # 得到一元势（负对数概率）
    U = unary_from_labels(labels, n_labels, gt_prob=0.5, zero_unsure=None)
    d.setUnaryEnergy(U)

    # 这将创建与颜色无关的功能，然后将它们添加到CRF中
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=8, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # 这将创建与颜色相关的功能，然后将它们添加到CRF中
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)

    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    MAP = colorize[MAP, :]
    imwrite(CRF_image_path, MAP.reshape(img.shape))
