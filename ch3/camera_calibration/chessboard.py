# -*- coding: utf-8 -*-
import cv2
import numpy as np

#内角点数量
pattern_size = (9, 6)
width = 720
height = 540
chessboard = np.ones((height, width, 3), dtype=np.uint8) * 255

w, h = pattern_size
#格子数量要比内角点数量多1个
w += 1
h += 1
grid_w = width // (w + 2)
grid_h = height // (h + 2)
grid_size = min(grid_w, grid_h)
board_w = grid_size * w #确保格子尺寸是整像素数
board_h = grid_size * h
start_w = (width - board_w) // 2 #居中显示
start_h = (height - board_h) // 2

for i in range(h):
    for j in range(w):
        if (i + j) % 2 == 0:
        	y = start_h + i * grid_size
        	x = start_w + j * grid_size
        	chessboard[y:y+grid_size, x:x+grid_size, :] = 0

cv2.imshow("chessboard", chessboard)
cv2.waitKey(0)

