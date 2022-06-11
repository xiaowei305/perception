import numpy as np
import math
import matplotlib.pyplot as plt


def dft(x):
    N = len(x)
    fourier = np.zeros((len(x), 2))
    two_pi = 2. * math.pi
    for k in range(N):
        for n in range(N):
            fourier[k, 0] += x[n] * math.cos(two_pi * k * n / N)
            fourier[k, 1] += x[n] * math.sin(-two_pi * k * n / N)

    magnitude = np.sqrt(fourier[..., 0] ** 2 + fourier[..., 1] ** 2)
    phase = np.arctan2(fourier[..., 1], fourier[..., 0])  # arctan(b/a)
    return magnitude, phase


if __name__ == "__main__":
    x = np.arange(100)
    freq1 = 0.1
    freq2 = 0.3
    y1 = np.sin(x * 2 * math.pi * freq1)  # 频率为0.1的正弦波
    y2 = np.sin(x * 2 * math.pi * freq2 + math.pi)  # 频率为0.3的正弦波, 相位为pi
    y = y1 + y2  # 两个不同频率的波叠加
    magnitude, phase = dft(y)  # 傅里叶变换得到频率的振幅和相位
    magnitude = magnitude * 2 / len(x)  # 对齐振幅的数量级
    frequency = x / len(x)  # 对齐频率的数量级
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    left = len(x) // 2  # 傅里叶变换的后半部分代表“负频率”与前半部分对称，可视化时可以忽略掉
    plt.plot(frequency[:left], magnitude[:left])
    plt.plot(frequency[:left], phase[:left])
    plt.show()