import numpy as np
arr = np.array([2,6,1,9,10,3,27])
index = np.where((arr >= 5) & (arr <=10))
print([arr[index]])
