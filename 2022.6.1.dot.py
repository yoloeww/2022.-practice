import numpy as np
m1 = np.array([[1,2,3],[4,5,6]])
print(m1)
m2 = np.array([[8,9,10,11],[12,13,14,15],[16,17,18,19]])
print(m2)
m3 = m1.dot(m2)
print(m3)


for i in range (1):
    for j in range (2):
        for k in range (3):
            m3[i][k] = m1[i][j]*m2[j][k]
print(m3)
