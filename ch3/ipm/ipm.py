import json
import cv2
import matplotlib.pyplot as plt
import numpy as np


def rotation_from_euler(roll=0., pitch=0., yaw=0.):
    sr, sp, sy = np.sin(roll), np.sin(-pitch), np.sin(yaw)
    cr, cp, cy = np.cos(roll), np.cos(-pitch), np.cos(yaw)
    R1 = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])
    R2 = np.array([
        [cp, 0, -sp],
        [0, 1, 0],
        [sp, 0, cp]
    ])
    R3 = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    R = np.identity(4)
    R[:3, :3] = R1 @ R2 @ R3
    return R


def translation_matrix(vector):
    M = np.identity(4)
    M[:3, 3] = vector[:3]
    return M


def load_camera_params(file):
    with open(file, 'rt') as handle:
        p = json.load(handle)

    fx, fy = p['fx'], p['fy']
    u0, v0 = p['u0'], p['v0']

    pitch, roll, yaw = p['pitch'], p['roll'], p['yaw']
    x, y, z = p['x'], p['y'], p['z']

    # Intrinsic
    K = np.array([[fx, 0, u0],
                  [0, fy, v0],
                  [0, 0, 1]])

    # Extrinsic
    R_veh2cam = np.transpose(rotation_from_euler(roll, pitch, yaw))
    T_veh2cam = translation_matrix((-x, -y, -z))

    # Rotate to camera coordinates
    R = np.array([[0., 1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])
    RT = R @ R_veh2cam @ T_veh2cam
    R = RT[:3, :3]
    T = RT[:3, 3:]
    return K, R, T


def get_homography(K, R, T):
    RT1 = np.concatenate((R[:, :2], T), axis=1)
    Hi = K @ RT1
    H = np.linalg.inv(Hi)
    return H


def ipm(coord, K, R, T):
    H = get_homography(K, R, T)
    xyz = H @ coord
    xyz = xyz / xyz[2, 0]
    return xyz[:, 0]


def draw_ipm(image, K, R, T):
    # Compute projection matrix
    H = get_homography(K, R, T)

    K_top = np.array([
            [10, 0, 0],
            [0, 10, 250],
            [0, 0, 1]])

    H = K_top @ H
    # Warp the image
    warped = cv2.warpPerspective(image, H, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    return warped


if __name__ == '__main__':
    # Retrieve camera parameters
    image = cv2.cvtColor(cv2.imread('stuttgart_01_000000_003715_leftImg8bit.png'), cv2.COLOR_BGR2RGB)
    TARGET_H, TARGET_W = 500, 500
    K, R, T = load_camera_params('camera.json')

    # Warp the image
    warped = draw_ipm(image, K, R, T)

    # roi = cv2.selectROI("please select a car.", image, False, False)
    roi = (823, 410, 121, 86)
    x, y, w, h = roi
    xc = int(x + w * 0.5)
    yc = y + h
    bottom_center = np.array([[xc], [yc], [1.0]])
    depth, dy, _ = ipm(bottom_center, K, R, T)
    print(depth, dy)

    cv2.circle(image, (xc, yc), 3, (255, 0, 0), 3)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(image, f"depth={depth:.3f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Draw results
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(image)
    ax[0].set_title('Front View')
    ax[1].imshow(warped)
    ax[1].set_title('IPM')
    plt.tight_layout()
    plt.show()
