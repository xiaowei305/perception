# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

chess_image_dir = 'images/'
square_size = 1.0    #每个棋盘格的实际尺寸，需要用尺子量出，这里先用1.0

pattern_size = (9, 6)  #内角点个数
w = pattern_size[0]
h = pattern_size[1]

pattern_points = np.zeros((h, w, 3), np.float32) #生成内角点世界坐标系，第一个内角点为原点
pattern_points[:, :, 0] = np.arange(w).reshape((1, w))
pattern_points[:, :, 1] = np.arange(h).reshape((h, 1))
pattern_points = pattern_points.reshape(-1, 3) * square_size

obj_points = []
img_points = []

img_names = os.listdir(chess_image_dir)
height, width = cv2.imread(os.path.join(chess_image_dir, img_names[0]), 0).shape[:2]

for fn in img_names: #每张棋盘格图片
    fpath = os.path.join(chess_image_dir, fn)
    img = cv2.imread(fpath, 0)
    found, corners = cv2.findChessboardCorners(img, pattern_size) #找到棋盘格内角点粗略位置
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)    #找到更精细的内角点位置
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        # 下面的代码是可视化棋盘格角点
        # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # cv2.drawChessboardCorners(vis, pattern_size, corners, found)  #将内角点画出来
        # cv2.imshow("corner", vis)

#相机标定
#rms是误差，K是内参，distortion是畸变，R, T分别是外参的旋转和平移
rms, K, distortion, R, T = cv2.calibrateCamera(
    obj_points,  # 世界坐标下的点的位置
    img_points,  # 图像中点的位置
    imageSize=(width, height),  # 图像尺寸
    cameraMatrix=None,  # 内参矩阵，如果不为None就是使用已知内参
    distCoeffs=None)    # 畸变参数，如果不为None就是使用已知畸变
print("RMS: %f" % rms)
print("intrinsic:\n" + str(K))
print("distortion:\n" + str(distortion))
print("R = \n", cv2.Rodrigues(R[0])[0])
print("T = \n", T[0])

np.savetxt("intrinsic.txt", K, fmt="%g")   #内参保存成文件
np.savetxt("distortion.txt", distortion.ravel(), fmt="%g") #畸变参数保存成文件
