from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
data = pd.DataFrame(cro_var, index=list("210"), columns=list("012"))
sns.heatmap(data, fmt='.2f', annot=True, annot_kws={'size': 15, 'weight': 'bold'})
plt.xlabel("hx", fontsize=15)
plt.ylabel("hy", fontsize=15)
plt.title("$\it{γ}$$_v$$_u$", fontsize=15)
plt.savefig(fname="./12γvu.svg", dpi=100)