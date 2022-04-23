## 3D目标检测

### camera_calibration
相机标定，您可以运行
```commandline
python3 chessboard.py
```
显示一张棋盘格，并用手机拍摄多个角度的图片（至少3张, 最好10张），并使用
```commandline
python calibrate.py
```
得到相机内参、外参和畸变参数。运行
```commandline
python undistort.py
```
可以对相机图片进行畸变矫正。
### ipm
逆透视变换，将路面上的点投影到俯视图（BEV）视角，并计算距离。

### lidar_to_image
将lidar点云投影到图像上，用于深度估计。

### 其他代码地址
DETR3D: https://github.com/WangYueFt/detr3d

Lift,Splat,Shoot: https://github.com/nv-tlabs/lift-splat-shoot

sfmlearner: https://github.com/JiawangBian/SC-SfMLearner-Release

monodepth2: https://github.com/nianticlabs/monodepth2