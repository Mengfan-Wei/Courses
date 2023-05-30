from numpy import *
import pandas as pd
from math import *

value = array([477, 696, 227, 646, 606, 791, 783])
distance = pd.read_excel('./dis.xlsx')
m = 600
# calculate covariance matrix
print(distance.shape)
ch = zeros(distance.shape)
print(ch)
for i in range(8):
    for j in range(8):
        ch[i][j] = 10 * exp(-0.4 * abs(distance[i][j]))
c = ch[1:, 1:]
c0 = ch[1:, 0]

d = dot(c0, linalg.inv(c))  # calculate the weight matrix
z0 = m + sum(dot(d, (value - m)))
err = 10 - sum(dot(d, c0))  # calculate the estimation error
# print(ch)
# print('*'*20)
# print(c)
# print('*'*20)
# print(c0)


# print('*'*20)
# print(d)
# print(z0)
# print(err)