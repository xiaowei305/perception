import cv2
import numpy as np

class HOG():
    def __init__(self, winSize):
        self.winSize = winSize
        self.blockSize = (16, 16)
        self.blockStride = (8, 8)
        self.cellSize = (8, 8)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize,self.blockStride,
                                     self.cellSize, self.nbins)

    def detect(self, image):
        winStride = self.winSize
        hist = self.hog.compute(image, winStride, padding = (0, 0))
        return hist

    def show(self, image, feature):
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        feature = feature.reshape(w, h, 4, 9).sum(axis=2)
        grid = 16
        harf_grid = grid // 2
        img = cv2.resize(image, (w * grid, h * grid))
        for i in range(w):
            for j in range(h): 
                for k in range(9):
                    x = int(10 * feature[i, j, k] * np.cos(np.pi / 9 * k))
                    y = int(10 * feature[i, j, k] * np.sin(np.pi / 9 * k))
                    #cv2.rectangle(img, (j * grid, i * grid), ((j+1) * grid, (i+1) * grid), (0, 255, 255))
                    x1 = i * grid + harf_grid - x
                    y1 = j * grid + harf_grid - y
                    x2 = i * grid + harf_grid + x
                    y2 = j * grid + harf_grid + y
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1) 
        cv2.imshow("img", img)
        cv2.waitKey(0)


hog = HOG(winSize=(64, 128))
image = cv2.imread("ped.jpg")
image_resize = cv2.resize(image, (64, 128))
feature = hog.detect(image_resize)
print(feature.size)
hog.show(image, feature)

