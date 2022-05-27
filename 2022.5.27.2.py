list1 = [1,2,3,4,3,2]   
for i in range (len(list1)):
          for j in range (len(list1)):
             if list1[i] == list1[j-i]:
                break;
             print(list1[i])
