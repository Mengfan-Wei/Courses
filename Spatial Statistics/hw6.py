from numpy import *
import pandas as pd
from math import *
import seaborn as sns
import matplotlib.pyplot as plt


def question1():
    value = array([477, 696, 227, 646, 606, 791, 783])
    distance = pd.read_excel('./dis.xlsx')
    ch = zeros(distance.shape)
    m = ch.shape[0]

    # calculate covariance matrix C(h)
    for i in range(m):
        for j in range(m):
            ch[i][j] = 10 * exp(-0.05 * abs(distance[i][j]) ** 2)
    c_ki = ch[1:, 1:]
    c_ki = insert(c_ki, 7, ones(c_ki.shape[1]), axis=0)
    c_ki = insert(c_ki, 7, ones(c_ki.shape[0]), axis=1)
    c_ki[7, 7] = 0
    c_k0 = ch[1:, 0]
    c_k0 = append(c_k0, [1])

    d = dot(linalg.inv(c_ki), c_k0)  # calculate the weight matrix
    z0 = sum(dot(value, d[:7]))  # calculate V0
    err = 10 - sum(dot(d[:7], c_k0[:7])) - d[7]  # calculate the estimation error
    print(z0)
    print(err)


def question2():
    value = array([[414, 396, 156]])
    print(value.shape)
    dist_ij = array([[0.0, 5.0, 5.5],
                     [5.0, 0.0, 5.0],
                     [5.5, 5.0, 0.0]])
    dist_ia = array([4.0, 3.5, 4.0])
    dist_ib = array([4.5, 2.0, 4.5])
    gamma_ij = ones((4, 4))
    for i in range(3):
        for j in range(3):
            gamma_ij[i][j] = 10 * (1 - exp(-0.3 * abs(dist_ij[i][j])))
    gamma_ij[3, 3] = 0
    gamma_ia = ones((4, 1))
    gamma_ib = ones((4, 1))
    for i in range(3):
        gamma_ia[i] = 10 * (1 - exp(-0.3 * abs(dist_ia[i])))
        gamma_ib[i] = 10 * (1 - exp(-0.3 * abs(dist_ib[i])))

    # Kriging first,average second
    # calculate the weight matrix
    lambda_a = dot(linalg.inv(gamma_ij), gamma_ia)  # (4,4),(4,1).-1
    lambda_b = dot(linalg.inv(gamma_ij), gamma_ib)
    # calculate z0
    print(lambda_a[:3].shape)
    za = sum(dot(lambda_a[:3], value))  # (3,),(3,1)
    zb = sum(dot(lambda_b[:3], value))
    z0 = (za + zb) / 2

    # Average first, kriging second
    gamma_iA = (gamma_ia + gamma_ib) / 2
    lambda_A = dot(linalg.inv(gamma_ij), gamma_iA)
    z02 = sum(dot(lambda_A[:3], value))
    print(z0)
    print(z02)


def calculate_varigram():
    '''
    计算半方差。输入：task3输出的二列数组（这个可以自己造个2列数组先写，能算就行）。输出：
    半方差
    '''
    v = array([1, 2, 3, 2, 3, 4, 2, 4, 6]).reshape(3, 3)
    u = array([4, 5, 6, 6, 7, 8, 7, 8, 9]).reshape(3, 3)
    h_list = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    cov = cor = var = cro_var = zeros((3, 3))  # Store covariance, correlogram, variogram and cross variogram
    for h in h_list:
        N = 0  # count of pairs
        vi, vj, ui, uj = [], [], [], []
        for i in range(0, 3):
            if i + h[0] > 2:
                break
            for j in range(0, 3):
                if j + h[1] > 2:
                    break
                vi.append(v[j, i])
                vj.append(v[j + h[1], i + h[0]])
                ui.append(u[j, i])
                uj.append(u[j + h[1], i + h[0]])
                N += 1
        vi_arr, vj_arr, ui_arr, uj_arr = array(vi), array(vj), array(ui), array(uj)
        cov[2 - h[0], h[1]] = mean(vi_arr * vj_arr) - mean(vi) * mean(vj)  # covariance
        cor[2 - h[0], h[1]] = cov[2 - h[0], h[1]] / (mean(vi_arr ** 2) - mean(vi) ** 2) ** 0.5 / (
                mean(vj_arr ** 2) - mean(vj) ** 2) ** 0.5  # correlogram
        cor[0, 2] = 1.0
        var[2 - h[0], h[1]] = sum((vi_arr - vj_arr) ** 2) / 2 / N  # variogram
        cro_var[2 - h[0], h[1]] = sum((vi_arr - vj_arr) * (ui_arr - uj_arr)) / 2 / N  # cross variogram
    # Thermal map
    # data = pd.DataFrame(cro_var, index=list("210"), columns=list("012"))
    # sns.heatmap(data, fmt='.2f', annot=True, annot_kws={'size': 15, 'weight': 'bold'})
    # plt.xlabel("hx", fontsize=15)
    # plt.ylabel("hy", fontsize=15)
    # plt.title("$\it{γ}$$_v$$_u$", fontsize=15)
    # plt.savefig(fname="./12γvu.svg", dpi=100)

    return var


def fit_variogram():
    '''
    二列数组，分别为h（距离）
    与其对应的半方差（这个可以自己造个2列数组先写，能算就行）。输出：拟合参数
    '''
    from scipy.optimize import curve_fit
    import numpy as np

    # 模拟生成一组实验数据
    h = np.arange(0, 100, 0.2)
    var = 10 + 5 * (1 - np.exp(-0.3 * h))
    noise = np.random.uniform(0, 0.1, len(h))
    var += noise

    # The model to be fitted
    def func(x, c0, c1, a):
        return c0 + c1 * (1 - np.exp(-x / a))

    # Initialize the parameters to be fitted
    c0 = min(var)
    c1 = max(var) - min(var)
    a = h[round(len(h) / 2)]
    init_para = [c0, c1, a]
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


if __name__ == '__main__':
    # question1()
    # question2()
    fit_variogram()
