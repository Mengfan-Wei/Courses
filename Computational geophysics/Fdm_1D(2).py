import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

dt = 0.002  # 时间步长
dx = 6  # x,z方向上的步长
nx = 200  # 采样点数
nt = 1000  # 采样时间点数
c = 3000  # 声波速度
f = 30  # 震源频率
# gama = 3  #频带控制参数
A = (c * dt / dx) ** 2

# 不稳定情况

# x,t的数组
x_vertor = np.linspace(0, (nx - 1) * dx, nx)
t_vertor = np.linspace(0, (nt - 1) * dt, nt)

# 初始化数组
p = np.zeros((nx, nt))
pold = np.zeros(nx)
p_t = np.zeros(nx)

d2p = np.zeros(nx)
end = len(p_t) - 1

# 有限差分计算
for tstep in range(nt - 1):
    # d2p = list(p_t[2:end + 1] - 2 * p_t[1:end] + p_t[0:end - 1])
    d2p[1:end] = p_t[2:end + 1] - 2 * p_t[1:end] + p_t[0:end - 1]

    # 固定边界条件
    d2p[0] = 0
    d2p[end] = 0
    # d2p = np.array([0] + d2p + [0])
    # 自由边界条件
    # d2p = np.array([d2p[1]] + d2p + [d2p[end - 2]])

    pnew = 2 * p_t - pold + A * d2p

    # pnew[100] += (dt ** 2) / dx * math.exp(-(2 * math.pi * f * (tstep - 0.05 / dt)\
    #                  * dt )**2 ) * math.cos(2 * math.pi * f * (tstep - 0.05 / dt) * dt)
    pnew[100] += (dt ** 2) / dx * (1 - 2 * (math.pi * f * dt * (tstep - 40)) ** 2) * \
                 math.exp(-(math.pi * f * dt * (tstep - 40)) ** 2)
    """ pold = p[:, tstep]
    p[:, tstep] = pnew
    p_t = pnew """

    pold = p_t
    p_t = pnew
    p[:, tstep] = pnew  # 输出数据

    # if tstep % 50 == 0:
    plt.clf()
    plt.plot(x_vertor, p_t)
    plt.xlabel("Distance (x), t = {:.3f}(s)".format(t_vertor[tstep]))
    plt.ylabel("p")
    # plt.axis((0,600, -0.0005, 0.0005))
    plt.pause(0.001)

plt.show()
# df=pd.DataFrame(p)
# df.to_csv('D:/VScode_python/fdm_1D.csv',float_format='%.12f')
