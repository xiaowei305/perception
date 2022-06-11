import numpy as np
import matplotlib.pyplot as plt


N = 100
signal = np.zeros(N)
signal[30] = 9.0
signal[70] = 7.5
noise = np.random.normal(loc=4.0, scale=2.0, size=N)
index = np.arange(N)
signal = signal + noise

plt.plot(index, signal)

refer_cell = 10
guard_cell = 2
alpha = 2.5

thresh = np.zeros(N)
for i in range(N):
    left_start = max(0, i - guard_cell - refer_cell)
    left_end = max(0, i - guard_cell)
    right_start = min(N, i + guard_cell)
    right_end = min(N, i + guard_cell + refer_cell)
    thresh[i] = alpha * np.concatenate((signal[left_start:left_end], signal[right_start:right_end])).mean()

plt.plot(index, thresh)
plt.show()
