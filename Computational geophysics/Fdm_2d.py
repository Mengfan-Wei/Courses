import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

dt = 0.001  # 时间步长
h = 6  # x,z方向上的步长
nx = 100  # 采样点数
nz = 100  # 采样点数
nt = 2000  # 采样时间点数
c = 3000  # 声波速度
f = 30  # 震源频率
gama = 3  # 频带控制参数
A = (c * dt / h) ** 2

p = np.zeros((nx, nz))
pold = np.zeros((nx, nz))
pnew = np.zeros((nx, nz))
# 二维有限差分
for k in range(1, nt):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            pnew[i][j] = A * (p[i + 1][j] + p[i - 1][j] + p[i][j + 1] + p[i][j - 1] - 4 * p[i][j]) \
                         - pold[i][j] + 2 * p[i][j]

            # 在中心点处设置一个Ricker子波

    pnew[50][50] += math.exp(-(2 * math.pi * f * (k - 0.035 / dt) \
                               * dt / gama) ** 2) * math.cos(2 * math.pi * f * (k - 0.035 / dt) * dt) / (h ** 2)
    # pnew[i][j] += math.exp(-(2 * math.pi * f * (k - 0.035 / dt)\
    #      * dt / gama)**2 ) * math.cos(2 * math.pi * f * (k - 0.035 / dt) * dt)

    # 边界条件
    # pnew[0, :] = pnew[1, :]
    # pnew[nx - 1, :] = pnew[nx - 2, :]
    pnew[0, :] = 0
    pnew[nx - 1, :] = 0

    # pnew[:, 0] = pnew[:, 1]
    # pnew[:, nz - 1] = pnew[:, nz - 2]
    pnew[:, 0] = 0
    pnew[:, nz - 1] = 0

    pold = p
    p = pnew

    # 输出数据
    """ df=pd.DataFrame(p)
    df.to_csv('D:/VScode_python_PC/fdm_2D.csv', float_format='%.5f') """

    # 画二维图像
    if k % 20 == 0:
        # plt.figure()
        plt.xlabel('x      t = {:.3f}(s)'.format(k * dt))
        plt.ylabel('z')

        plt.imshow(p, cmap=plt.cm.jet)
        # plt.imshow(p, cmap=plt.cm.gray,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        plt.pause(0.5)

    # 画一维图像
    '''if k % 20 ==0:
        x_vertor = np.arange(0, 600, 6)
        p_vertor = p[:, 50]
        plt.clf()
        plt.plot(x_vertor, p_vertor)
        plt.xlabel("Distance (x), t = {:.3f}(s)".format(k * dt))
        plt.ylabel("p")
        # plt.axis((0, 600, -0.005, 0.005))
        plt.pause(0.5)   '''

plt.show()
