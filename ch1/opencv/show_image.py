#_*_coding:utf8_*_
import cv2

image = cv2.imread("lenna.jpg")
print(image.shape)
# print(image)  # 你可以试试将数值打印出来
cv2.imshow("image", image)
cv2.waitKey(0)  # 如果不加这句，图片会一闪而过

