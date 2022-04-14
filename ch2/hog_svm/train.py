# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys

image_dir = "dataset/images/"
label_dir = "dataset/labels/"
negative_dir = "dataset/negative/"

winSize = (64, 128)   #HOG窗口大小（像素）
blockSize = (16, 16)  #HOG块大小（像素）
blockStride = (8, 8)  #HOG块的滑动步长（像素）
cellSize = (8, 8)     #胞元大小（像素）
nbins = 9             #180度分成多少个直方图

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

def get_hog(img):
    '''
    计算HOG特征，输出长度为3780的数组
    计算之前先将图片缩放到64*128，保证刚好一个窗口的大小
    '''
    resized = cv2.resize(img, (64, 128))
    winStride = (32, 64) #滑动窗口步长，注意不要和块的步长混淆
    hist = hog.compute(resized, winStride, padding = (0, 0))
    return hist.flatten() #HOG特征转为一维数组

def iou(a, b):
    '''
    计算两个框的IOU
    '''
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    if yy2 < yy1 or xx2 < xx1:
        return 0
    area_iou = (xx2 - xx1) * (yy2 - yy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    iou_value =  float(area_iou) / (area_a + area_b - area_iou)
    return iou_value

positive = []
negative = []

def is_negative(box, boxes, w, h):
    '''
    如果生成的框，与行人框的IOU>0.3或者超出图像边界，则不加入负样本训练集
    '''
    if box[0] < 0 or box[1] < 0 or box[2] >= w or box[3] >= h:
        return False
    for b in boxes:
        if iou(box, b) > 0.45:
            return False
    return True

for img in os.listdir(image_dir):
    image = cv2.imread(os.path.join(image_dir, img))
    label_file = os.path.join(label_dir, img.replace(".png", ".txt"))
    boxes = []
    f =  open(label_file)
    while True: 
        line = f.readline()
        if line == '':
            break
        x1, y1, x2, y2 = map(int, line.split(","))
        boxes.append((x1, y1, x2, y2))

    for box in boxes:
        x1, y1, x2, y2 = box
        pedestrain = image[y1:y2, x1:x2, :]  #原图计算HOG
        positive.append(get_hog(pedestrain))
        pedestrain = cv2.flip(pedestrain, 1) #水平翻转后也当作正样本
        positive.append(get_hog(pedestrain))

        background = []
        w = x2 - x1
        shift_w = int(w * 0.6)
        h = y2 - y1
        shift_h = int(h * 0.6)

        #分别将原始框左移、右移、上移、下移生成负样本
        background.append((x1+w, y1, x2+w, y2))
        background.append((x1-w, y1, x2-w, y2))
        background.append((x1+shift_w, y1, x2+shift_w, y2))
        background.append((x1-shift_w, y1, x2-shift_w, y2))
        background.append((x1, y1 + shift_h, x2, y2 + shift_h))
        background.append((x1, y1 - shift_h, x2, y2 - shift_h))
        background.append((x1, y1, x1+shift_w, y1 + shift_h))
        background.append((x1+shift_w, y1 + shift_h, x2, y2))

        for box in background:
            if is_negative(box, boxes, image.shape[1], image.shape[0]):
                x1, y1, x2, y2 = box
                no_ped = image[y1:y2, x1:x2, :]
                negative.append(get_hog(no_ped))

# 另一种负样本，图片中不包含行人，固定取左下角120*240像素作为负样本
for img in os.listdir(negative_dir):
    image = cv2.imread(os.path.join(negative_dir, img))
    image = image[image.shape[0] - 240:, :120, :]
    negative.append(get_hog(image))
    image = image[:240:, :120, :]
    negative.append(get_hog(image))

npos = len(positive)
nneg = len(negative)
num = npos + nneg

print("positive samples: %d" % npos)
print("negative samples: %d" % nneg)
label = np.ones((num, 1), dtype=np.int32)
label[npos:, 0] = -1

data = np.array(positive + negative).reshape(num, -1)
print("input data shape: " + str(data.shape))
print("input label shape: " + str(label.shape))

# opencv2和3的用法略有不同:
if cv2.__version__[0] == '2':
  svm_params = dict(kernel_type = cv2.SVM_LINEAR,  # 线性核函数
                    svm_type = cv2.SVM_C_SVC,      # 分类SVM
                    C=0.015)                       # 松弛变量
  svm = cv2.SVM()
else:
  svm = cv2.ml.SVM_create()
  svm.setKernel(cv2.ml.SVM_LINEAR)  # 线性核函数
  svm.setType(cv2.ml.SVM_C_SVC)     # 分类SVM
  svm.setC(0.015)                   # 松弛变量

svm.train(data, cv2.ml.ROW_SAMPLE, label)  
svm.save('svm.xml')
