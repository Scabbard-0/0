import numpy as np
import math
import matplotlib.pyplot as plt

SAMPLE_NUM = 1000  # 样点个数
M = 5  # 多项式阶数

# 产生带有高斯噪声的信号
mid, sigma = 0, 0.3  # 均值和方差
noise = np.random.normal(mid, sigma, SAMPLE_NUM).reshape(SAMPLE_NUM, 1)

# 产生SAMPLE_NUM个序号(范围是2pi)
x = np.arange(0, SAMPLE_NUM).reshape(SAMPLE_NUM, 1) / (SAMPLE_NUM - 1) * (2 * math.pi)

y = np.sin(x)
y_noise = np.sin(x) + noise

# 绿色曲线显示x - y，散点显示x - y_noise
plt.title("")
plt.plot(x, y, 'g', lw=4.0)
plt.plot(x, y_noise, 'bo')

X = x
for i in range(2, M + 1):
    X = np.column_stack((X, pow(x, i)))

X = np.insert(X, 0, [1], 1)

W = np.linalg.inv((X.T.dot(X)) + np.exp(-8) * np.eye(M + 1)).dot(X.T).dot(y_noise)
y_estimate = X.dot(W)

plt.plot(x, y_estimate, 'r', lw=4.0)
plt.show()
