import numpy as np
import matplotlib.pyplot as plt

# 波速
c = 1

# x范围和节点数
x_min, x_max, number_dx = 0, 1, 101
# t范围和节点数
t_min, t_max, number_dt = 0, 2, 401

# c = 2
# x_min, x_max, number_dx = 0, 1, 101
# t_min, t_max, number_dt = 0, 3, 451

# x,t的数组
x_vertor = np.linspace(x_min, x_max, number_dx)
print(x_vertor)
t_vertor = np.linspace(t_min, t_max, number_dt)

dx = (x_max - x_min) / (number_dx - 1)
dt = (t_max - t_min) / (number_dt - 1)

# 初始化数组p - [101,404]的矩阵 - 行索引0-100，列索引0-400
p = np.zeros((number_dx, number_dt))

# func = lambda x: np.sin(4 * np.pi * x)
# for i, x in enumerate(x_vertor):
#     # t = 0 时的初值
#     p[i][0] = func(x)

# 输入x,输出 ：后的0.5*sin(4pi*x)
func = lambda x: 0.5 * np.sin(4 * np.pi * x)
for i, x in enumerate(x_vertor):  # i获取索引，x获取值,i-[0,100],x-[0.0,1.0]
    # 当x在（0,0.25）或（0.5,0.75）时，对应节点（第i个节点）在0时刻的压力为0.5*sin(4pi*x)
    if (0 <= x <= 0.25) or (0.5 <= x <= 0.75):
        p[i][0] = func(x)
    else:
        p[i][0] = 0

# 初始化101个0元素的矩阵
pold = np.zeros(number_dx)
tstep = 0
# 0时刻所有节点的压力
p_t = p[:, tstep]
# 画出所有节点与0时刻对应压力的图像
plt.plot(x_vertor, p[:, tstep])
# x轴标题为Distance (x), t = x(s)   x从0.00变化到2.00
plt.xlabel("Distance (x), t = {:.2f}(s)".format(t_vertor[tstep]))
plt.ylabel("p")


plt.axis((x_min - x_max / 10, x_max + x_max / 10, -1.2 * np.max(p),
          1.2 * np.max(p)))
# 对于动态图像，间隔dt/2=0.005更新一次
plt.pause(dt / 1000)

end = len(p_t) - 1
# 遍历400个时间步
for tstep in range(number_dt - 1):
    d2p = list(p_t[2:end + 1] - 2 * p_t[1:end] + p_t[0:end - 1])
    d2p = np.array([0] + d2p + [0])
    pnew = 2 * p_t - pold + (c * dt / dx) ** 2 * d2p
    pold = p[:, tstep]
    p[:, tstep + 1] = pnew
    p_t = pnew
    plt.clf()
    plt.plot(x_vertor, p[:, tstep + 1])
    plt.xlabel("Distance (x), t = {:.2f}(s)".format(t_vertor[tstep + 1]))
    plt.ylabel("p")
    plt.axis((x_min - x_max / 10, x_max + x_max / 10, -1.5 * np.max(p),
              1.5 * np.max(p)))
    plt.pause(dt / 2000)

plt.show()
