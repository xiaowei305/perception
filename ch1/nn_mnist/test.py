#_*_coding:utf8_*_
import cv2
import numpy as np
import os
import random
from neural_network import nn

#手写数字识别demo, 其中mnist.npy需要通过运行train.py得到
net = nn()
net.load("mnist.npy") #加载权重
files = os.listdir("dataset/test")

random.shuffle(files)
for f in files:
    img = cv2.imread("dataset/test/" + f, cv2.IMREAD_GRAYSCALE)
    x = img.astype(np.float32).reshape(1, 1024)
    y = net.forward(x)                # 前向传播预测数字

    big_img = cv2.resize(img, (256, 256))     #原图是32 * 32太小，为方便展示，放大一下
    cv2.putText(big_img, str(y), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2) #在图像左上角将检测结果写进去
    cv2.imshow("window", big_img) #显示图片
    k = cv2.waitKey(0)
    if k == 27:
        break

