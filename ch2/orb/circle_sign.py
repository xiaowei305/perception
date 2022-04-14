#_*_coding:utf8_*_
import cv2
import numpy as np

def filter_matches(kp1, kp2, matches, ratio=0.75):
    '''
    将没有成功匹配或距离大于平均距离0.75倍的点去掉
    '''
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp1[m.trainIdx])
    p1 = [[int(kp.pt[0]), int(kp.pt[1])] for kp in mkp1]
    p2 = [[int(kp.pt[0]), int(kp.pt[1])] for kp in mkp2]
    return p1, p2

def draw_matches(template, image, kp1, kp2):
    '''
    画出匹配点
    '''
    h1, w1, _ = template.shape
    h2, w2, _ = image.shape
    out = np.zeros((h2, w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = template.copy()
    out[:, w1:] = image.copy()

    for ((x1, y1), (x2, y2)) in zip(kp1, kp2):
        cv2.line(out, (x1, y1), (w1 + x2, y2), (0, 0, 255), 1)

    cv2.imshow("matching", out)


if __name__ == "__main__":
    image = cv2.imread("road.jpg") 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("origin_image", image)

    # 使用霍夫变换检测圆，函数内部包含了Canny,因此不再需要先检测边缘
    # dp=2, 霍夫空间的分辨率是原图的1/2
    # minDist=100 圆心之间的最小距离
    # param1 和 param2分别是Canny算子的两个阈值
    # minRadius和maxRadius是圆的最小，最大半径
    circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT,
                               dp=2, minDist=100,
                               param1=150, param2=100,
                               minRadius=30, maxRadius=50)


    boxes = []
    image2 = image.copy()
    for circle in circles[0]:
        x, y, r = map(int, circle)
        cv2.circle(image2, (x, y), r, (0, 0, 255), 2)
        box = (x - r, y - r, x + r, y + r)  # 用圆心和半径得到包围框
        boxes.append(box)
        #cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)

    cv2.imshow("circle_detection", image2)

    # 接下来开始使用orb特征匹配
    template = cv2.imread("sign.jpg")  # 模板图像
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    height, width = template_gray.shape
    
    orb = cv2.ORB_create()  # orb 特征提取器，有很多参数我们全部使用默认值
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=1)
    search_params = dict(check=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # 使用Flann的LSH匹配算法
    
    keypoint = orb.detect(template_gray, None)  # 检测特征点的位置
    keypoint, desc = orb.compute(template_gray, keypoint) # 为特征点计算描述子

    image3 = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        sub_image = gray[y1:y2, x1:x2]
        sub_image = cv2.resize(sub_image, (width, height))
        keypoint2 = orb.detect(sub_image, None)
        keypoint2, desc2 = orb.compute(sub_image, keypoint2)
        matches = flann.knnMatch(desc, desc2, k=2)
        mp1, mp2 = filter_matches(keypoint, keypoint2, matches)  # 将没有匹配上的点过滤掉
        for pt in mp2:
            pt[0] = int(float(pt[0]) * (x2 - x1) / width) + x1   # 坐标从图像块转换为原图
            pt[1] = int(float(pt[1]) * (y2 - y1) / height) + y1
            # cv2.circle(image, (pt[0], pt[1]), 2, (0, 0, 255), 1)
    
        output = draw_matches(template, image, mp1, mp2)  # 显示匹配结果
        print(len(mp1))
        if len(mp1) > 20:  # 匹配的点数超过20就认为匹配成功, 输出框（也可以输出圆）
            cv2.rectangle(image3, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("image", image3)
    cv2.waitKey(0)
