from numpy import *
import numpy as np
import pandas as pd
from math import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_variogram(h, var):
    '''
    二列数组，分别为h（距离）
    与其对应的半方差（这个可以自己造个2列数组先写，能算就行）。输出：拟合参数
    '''

    # 模拟生成一组实验数据
    h = np.arange(0, 100, 0.2)
    var = 10 + 5 * (1 - np.exp(-0.3 * h))
    noise = np.random.uniform(0, 0.1, len(h))
    var += noise

    # The model to be fitted
    def func(x, c1, a):
        return c1 * (1 - np.exp(-x / a))

    # Initialize the parameters to be fitted
    c0 = min(var)
    c1 = max(var) - min(var)
    a = h[round(len(h) / 2)]
    init_para = [c1, a]
    para, cov = curve_fit(func, h, var, p0=init_para)
    y_fit = [func(a, *para) for a in h]

    fig, ax = plt.subplots()
    ax.scatter(h, var, color='b', marker='x', label='Data point')
    ax.plot(h, y_fit, 'g', label='Variogram model')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Variogram')
    plt.legend()
    plt.show()

    return para


def kriking(m_a_, n_, z_a_, c0, c1, a):
    '''
    输入：task5拟合的参数，task2输出的3列数组，
    估计区域的坐标（这个也可以自己造参数先写，能算就行）。输出：估计结果。
    '''
    estimation_coord = pd.read_excel('./coord.xlsx')
    ch = zeros(estimation_coord.shape)
    m = ch.shape[0]

    # The magnitude, number of earthquakes and focal depth
    # of the estimated region are calculated in turn
    for value in [m_a_, n_, z_a_]:
        size = len(value)
        # calculate covariance matrix C(h)
        for i in range(m):
            for j in range(m):
                ch[i][j] = c0 + c1 * (1 - exp(a * abs(estimation_coord[i][j])))
        c_ki = ch[1:, 1:]
        c_ki = insert(c_ki, size, ones(c_ki.shape[1]), axis=0)
        c_ki = insert(c_ki, size, ones(c_ki.shape[0]), axis=1)
        c_ki[size, size] = 0
        c_k0 = ch[1:, 0]
        c_k0 = append(c_k0, [1])

        d = dot(linalg.inv(c_ki), c_k0)  # calculate the weight matrix
        z0 = sum(dot(value, d[:size]))  # calculate V0
        err = 10 - sum(dot(d[:size], c_k0[:size])) - d[size]  # calculate the estimation error
        print(z0)
        print(err)


def basic_stat():
    from scipy import stats
    m_ = np.arange(10)
    m_mean = np.mean(m_)  # 平均数
    m_std = np.std(m_)  # 标准差
    m_std_div_mean = m_std / m_mean  # 标准差/平均数
    m_min = np.min(m_)
    m_max = np.max(m_)
    m_mode = stats.mode(m_)[0][0]  # 总数

    # Q1，平均数，Q3
    m_Q1, M, m_Q3 = np.percentile(m_, (25, 50, 75), interpolation='midpoint')

    print(m_)
    print(m_mean)
    print(m_std)
    print(m_std_div_mean)
    print(m_min)
    print(m_max)
    print(m_mode)
    print(m_Q1, M, m_Q3)


def inverse_distance(xx_, yy_, m_a_, x_estimate, y_estimate):
    # find values which is not nan
    xx_ = xx_.ravel()
    yy_ = yy_.ravel()
    m_a_ = m_a_.ravel()
    xx_obs = []
    yy_obs = []
    m_a_obs = []
    xx_obs.append(x_estimate)
    yy_obs.append(y_estimate)
    m_a_obs.append(np.nan)
    for i in range(xx_.shape[0]):
        if np.isnan(m_a_[i]):
            continue
        else:
            xx_obs.append(xx_[i])
            yy_obs.append(yy_[i])
            m_a_obs.append(m_a_[i])
    # calculate distance matrix
    distance_ = np.zeros((len(xx_obs), len(xx_obs)))
    for i_ in range(distance_.shape[0]):
        for j_ in range(distance_.shape[1]):
            distance_[i_][j_] = math.sqrt((xx_obs[i_] - xx_obs[j_]) ** 2 + (yy_obs[i_] - yy_obs[j_]) ** 2)

    #  estimate and error:
    m_a_obs = np.array(m_a_obs)
    # p=0->same weight to every samples
    # p=1->standard or local method
    for p in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        m_estimate = np.sum(m_a_obs[1:] / distance_[0, 1:] ** p) / np.sum(1. / (distance_[0, 1:] ** p))
        err = m_estimate - m_a_obs[0]


def is_2_order_stationary(m_a, distance_):
    d_list = np.linspace(1, 50, 100)
    mean_list = np.zeros_like(d_list)
    for d in d_list:
        id_value = np.where(distance_[0, 1:] < d)
        mean = np.mean(m_a[id_value])
        np.append(mean_list, mean)
    plt.scatter(d_list, mean_list, marker="x", )



if __name__ == '__main__':
    fit_variogram()
