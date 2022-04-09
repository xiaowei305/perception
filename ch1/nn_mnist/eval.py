#_*_coding:utf8_*_
import cv2
import numpy as np
import os
from neural_network import nn

#评估神经网络的精度
net = nn()
net.load("mnist.npy") #加载权重

files = os.listdir("dataset/test/")
true_positive = 0
for f in files:
    img = cv2.imread("dataset/test/" + f, cv2.IMREAD_GRAYSCALE)
    x = img.astype(np.float32).reshape(1, 1024)
    y = net.forward(x)    #预测
    if int(f[0]) == y:    #如果预测准确，true positive增加
        true_positive += 1
accuracy = float(true_positive) / float(len(files))
print("accuracy=%f" % accuracy) #精度=true_positive / 全部样本
