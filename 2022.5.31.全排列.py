list1 = [1,2,3]
result = []
for i in list1:
    for j in list1:
        for k in list1:
            if len(set((i,j,k))) == 3:
                   result.append(list((i,j,k)))
print(result)
