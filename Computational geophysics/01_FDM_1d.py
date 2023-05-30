import numpy as np
from matplotlib import pyplot as plt
import math

c = 300  # 波速
dx = 0.6  # 采样间隔
dt = 0.002  # 时间步长
nx = 100  # 采样点数
nt = 500  # 总时间步
f = 30  # 主频率

x = np.linspace(0, (nx - 1) * dx, nx)  # 采样点的坐标
t = np.linspace(0, (nt - 1) * dt, nt)  # 采样时刻
p = np.zeros(nx)  # p(t)
pold = np.zeros(nx)  # p(t-dt)
pnew = np.zeros(nx)  # p(t+dt)
d2p = np.zeros(nx)  # space derivative
t0 = 0.05

for i in range(1, nt):  # 迭代时间步
    print("Time step:", i)
    for j in range(2, nx - 1):  # 更新采样点
        d2p[j] = (p[j + 1] - 2 * p[j] + p[j - 1]) / dx ** 2  # space derivative
    pnew = 2 * p - pold + d2p * dt ** 2 * c ** 2  # time extrapolation
    pnew[50] += (dt ** 2) / dx * (1 - 2 * (math.pi * f * dt * (i - 40)) ** 2) * \
                 math.exp(-(math.pi * f * dt * (i - 40)) ** 2)  # 更新源（中心点）
    pold = p  # 更新时间
    p = pnew

    # 固定边界条件
    p[1] = 0
    p[nx - 1] = 0

    # display
    plt.clf()  # Clear the current figure.
    plt.plot(x, p, 'b')
    plt.title('FD')
    plt.xlabel("Distance (x), t = {:.3f}(s)".format(t[i]))  # Distance (x), t = x(s)   x从0.00变化到2.00
    plt.ylabel("p")
    plt.pause((0.001))  # 对于动态图像，每0.005更新一次
plt.show()
