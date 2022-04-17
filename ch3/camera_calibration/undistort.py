# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

image_dir = "images"

camera_matrix = np.loadtxt("intrinsic.txt") #加载内参
dist_coefs = np.loadtxt("distortion.txt")   #加载畸变参数
for fn in os.listdir(image_dir):
    name = fn[:fn.rfind('.')]
    img_file = os.path.join(image_dir, fn)
    img = cv2.imread(img_file)

    h, w = img.shape[:2]
    #获取畸变矫正矩阵，以及黑边的尺寸
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                      dist_coefs,
                                                      imageSize=(w, h),
                                                      alpha=1,  # alpha是图像缩放系数
                                                      )

    #畸变矫正
    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    #矫正过后会有黑边，裁掉黑边（也可以不裁掉）
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv2.imshow("origin", img)
    cv2.imshow("undistort", dst)
    k = cv2.waitKey(0)
    if k == ord('q') or k == 27:
        break
