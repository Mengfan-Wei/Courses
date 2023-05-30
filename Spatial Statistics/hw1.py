import numpy as np

A = np.array([2, 3, -1, 3, 4, 3, -1, 3, 7]).reshape(3, 3)
B = np.array([1, 2, -2, 1, 3, 2, -3, 1, 0]).reshape(3, 3)
AB = np.dot(A, B)
BA = np.dot(B, A)

print(A + B)
print(A - B)
print(AB)
print(BA)
print(np.linalg.det(A)) # 行列式

c = np.array([[1, 1],
              [1, -1]])
d = np.array([1, 0])
e = np.dot(np.linalg.inv(c), d) # 对C求逆，再和d进行矩阵乘法
