import numpy as np
a = np.array([[3,2,1],[1,1,1],[1,2,-1]])
b = np.array([-3,5,-2])
c = b.T
x = np.linalg.solve(a,b)

print(x)
