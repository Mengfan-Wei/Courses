from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt

def basic_stat():
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
    p_list = np.linspace(0, 1, 100)
    err_list = []
    for p in p_list:
        m_estimate = np.sum(m_a_obs[1:] / distance_[0, 1:] ** p) / np.sum(1 / distance_ ** p)
        err = m_estimate - m_a_obs[0]
        err_list.append(err)
    plt.plot(p_list,err_list)
    plt.xlabel("power of distance")
    plt.ylabel("error")
    plt.show()
