import os

import matplotlib.pyplot as plt
import cv2
import numpy as np


def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.split()])
    return data


def project_lidar_to_cam2(calib):
    # 雷达到相机的变换矩阵
    P_lidar2cam = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4),
                             np.array([0., 0., 0., 1.])))
    R0_camera = np.eye(4)    # 转换到0号相机的旋转矩阵，KITTI数据集特有
    R0_camera[:3, :3] = calib['R0_rect'].reshape(3, 3)
    P_lidar2cam = R0_camera @ P_lidar2cam
    K_cam2 = calib['P2'].reshape((3, 4)) # P2代表2号相机（左前）外参
    proj_mat = K_cam2 @ P_lidar2cam
    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    points = np.vstack((points, np.ones((1, num_pts))))  # 扩展xyz为齐次坐标
    points = proj_mat @ points
    points[:2, :] /= points[2, :]  # 归一化
    return points[:2, :]


def render_lidar_on_image(pcd, img, calib, img_width, img_height):
    proj_lidar2cam2 = project_lidar_to_cam2(calib)  # 从标定参数获得点云到相机2的变换矩阵

    pts_2d = project_to_image(pcd.transpose(), proj_lidar2cam2)  # 投影点云到图片

    # 过滤不在图像可见范围内的点(超出图像范围或者深度为负）
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pcd[:, 0] > 0))[0]

    imgfov_pc_pixel = pts_2d[:, inds]

    # 根据投影关系，找到原始点云的深度
    imgfov_pc_lidar = pcd[inds, :]
    imgfov_pc_lidar = np.hstack((imgfov_pc_lidar, np.ones((imgfov_pc_lidar.shape[0], 1))))
    imgfov_pc_cam2 = proj_lidar2cam2 @ imgfov_pc_lidar.transpose()  # 点云转换到相机坐标

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]   # 点云的第三个坐标z就是深度
        color = cmap[int(640.0 / depth), :]  # 将深度转换为颜色
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=-1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img


if __name__ == '__main__':
    rgb = cv2.cvtColor(cv2.imread(os.path.join('data/000114_image.png')),
                       cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    calib = read_calib_file('data/000114_calib.txt')   # 相机和激光雷达标定参数

    lidar_scan = np.fromfile('data/000114.bin', dtype=np.float32)
    lidar_scan = lidar_scan.reshape((-1, 4))[:, :3]  # 点云是4维的，xyz和反射强度

    render_lidar_on_image(lidar_scan, rgb, calib, img_width, img_height)  # 将点云投影到图像
