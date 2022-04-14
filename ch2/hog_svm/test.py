# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys

winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
winStride = (32, 32)

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

image_dir = "dataset/test"

is_opencv2 = False
if cv2.__version__[0] == '2':
    is_opencv2 = True

if is_opencv2:
  svm = cv2.SVM().load("svm.xml")
else:
  svm = cv2.ml.SVM_load('svm.xml')

def get_hog(img):
    hist = hog.compute(img, winStride, padding = (0, 0))
    return hist

def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = (ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh)
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    if yy2 < yy1 or xx2 < xx1:
        return 0
    area_iou = (xx2 - xx1) * (yy2 - yy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    iou_value =  float(area_iou) / float(area_a + area_b - area_iou)
    return iou_value

def nms(boxes):
    '''
    非最大抑制算法NMS
    一般来说需要对box进行排序，但SVM输出的box并没有分数，所以只能按照从前到后过滤
    '''
    for i, b1 in enumerate(boxes):
        if b1[2] == 0 or b1[3] == 0:
            continue
        for b2 in boxes[i+1:]:
            if b2[2] == 0 or b2[3] == 0:
                continue
            if iou(b1, b2) > 0.1:
                b2[2] = 0
                b2[3] = 0
    return [b for b in boxes if b[2] != 0 and b[3] != 0]

def detect_multi_scale(image):
    max_width = 150          #设置滑动窗口的最大宽度为150像素
    step = 10                #滑动窗口的大小每次减少10像素
    boxes = []
    for size in range(10):   #一共将滑动窗口减少10次，最后一个窗口是50像素
        win_w = max_width - step * size
        win_h = win_w * 2    #滑动窗口高度固定为宽度的2倍
        scale = float(winSize[0]) / win_w
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image2 = cv2.resize(image, (new_width, new_height))  #调整图像大小，使滑动窗口刚好为60 *128像素
        hists = get_hog(image2).reshape(-1, 3780)            #一次性计算所有滑动窗口的HOG特征
        wn = (new_width + 1) / winStride[0] - 1              #计算宽度方向上产生了多少个滑动窗口

        if is_opencv2:
            for idx, hist in enumerate(hists):
                _, (ret,) = svm.predict(hist)
                if ret > 0:
                    x = (idx % (wn)) * win_w * winStride[0] / winSize[0]  #根据窗口的位置算出窗口的偏移，从而恢复出左上角坐标
                    y = (idx / (wn)) * win_h * winStride[1] / winSize[1]  #由于宽度上产生了多少窗口已知，根据滑动偏移来计算
                    boxes.append([x, y, win_w, win_h]) 
        else:
           result, classes = svm.predict(hists)                 #所有滑动窗口产生的HOG特征全部送进SVM
           for idx, cls in enumerate(classes):                  #遍历分类结果，找到分类为1的结果（>0)
               if cls > 0:
                   x = (idx % (wn)) * win_w * winStride[0] / winSize[0]  #根据窗口的位置算出窗口的偏移，从而恢复出左上角坐标
                   y = (idx / (wn)) * win_h * winStride[1] / winSize[1]  #由于宽度上产生了多少窗口已知，根据滑动偏移来计算
                   boxes.append([x, y, win_w, win_h]) 
        filtered = nms(boxes)   #NMS后处理过滤结果
    return filtered

for img in os.listdir(image_dir):
    image = cv2.imread(os.path.join(image_dir, img))
    boxes = detect_multi_scale(image)
    for b in boxes:
        x, y , w, h = map(int, b)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("window", image)
    if cv2.waitKey(0) == 27:
        break
