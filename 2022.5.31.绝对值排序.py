import math
list1 = [-21 , -12,  5, 9, 36]
list2=[]
n= len(list1)
for i in range (n):
     list2.append(abs(list1[i]))
list2.sort()
print(list2)
