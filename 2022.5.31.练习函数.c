a = "asds23199  "
num = 0
space = 0
b = 0
for i in range(len(a)):
    if (a[i].isdigit()):
        num = num + 1
    elif (a[i].isspace()):
        space = space + 1
    else:
        b = b + 1
print(num)
print(space)
print(b)
