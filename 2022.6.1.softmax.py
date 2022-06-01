import numpy as np
m1 = np.random.randn(1,3)
m2 = []
print(m1)
m2=np.exp(m1[:])
m3 = m2.sum()
print(m3)
m4 = m2/m3
print(m4)
