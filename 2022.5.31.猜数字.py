import random
n = input()
for i in range (5):
    value = random.randint(1,100)
    if(i == value):
        print("right")
        break;
    else:
        print("again")
        n=input()
