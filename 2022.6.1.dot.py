import numpy as np
m1 = np.array([[1,2,3],[4,5,6]])
#print(m1)
m2 = np.array([[8,9,10,11],[12,13,14,15],[16,17,18,19]])
#print(m2)
m3 = m1.dot(m2)
print(m3)
m = len(m1)
n = len(m2[0])
m3 = [[0 for i in range(n)] for j in range(m)]

for i in range (m):
    for j in range (n):
        for k in range (len(m2)):
            m3[i][j] += m1[i][k]*m2[k][j]
print(m3)
