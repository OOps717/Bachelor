# Dynamically
x = int(input())
j = 1
for i in range(2,x+1):
  j = j*i
eval(str(j))

# Recurcively
def fact (x):
  if (x != 1):
    x = x * fact (x-1)
  return x


print (fact(int(input())))