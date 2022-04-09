#_*_coding:utf8_*_
import cv2
import numpy as np
import os
from neural_network import nn

# 运行前请解压dataset.zip得到training_img和test_img
def get_training_data(dataset_dir):
    images = []
    labels = []
    train_dir = os.path.join(dataset_dir, "train")
    files = os.listdir(train_dir)   #获取训练数据列表
    np.random.shuffle(files)
    for f in files:                      #读取每一张图片，并加入到样本中
        img_path = os.path.join(train_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #以灰度图的格式读取
        img = img.flatten()              #默认读取的是32 * 32, 展开成1024维
        images.append(img)               #样本加入list中
        num = int(f[0])                  #文件名的第一个字符就是标签，这里需要转成int型, 例如8_001.jpg是数字8
        label = np.zeros(10)
        label[num] = 1                   #标签是以one-hot方式存放的，举例[0,0,1,0,0,0,0,0,0,0]代表2
        labels.append(label)             #图片对应标签也加入到list
    return (np.float32(images), np.float32(labels)) #转换为numpy 的数组用于训练

images, labels = get_training_data("dataset")         #获取训练数据
net = nn(1024, 16, 10) 
net.train(images, labels)
net.save("mnist.npy")
