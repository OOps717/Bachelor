import numpy as np
import matplotlib.pyplot as plt
import math 
import time
import random

#Trapezoidal
def trapezoidalCircle(a,b,n):
    delta = (b - a)/n
    list = [a]
    i = a + delta
    while i <= b:
        list.append(i)
        i = i + delta
    area = 0
    for j in list:
        if j==list[0] or j==list[-1]:
            area = area + delta/2 * (np.sqrt(b**2 - j**2))
        else:
            area = area + delta * (np.sqrt(b**2 - j**2))
    return 2*area

time_start = time.clock()
trapezoidalCircle(-1,1,100000)
time_elapsed1 = (time.clock() - time_start)

print('Trapezoidal:')
listN = [10,100,1000,10000,100000]
for i in listN:
    print('n = ',i,',pi = ',trapezoidalCircle(-1,1,i))

#Stochastic method
def randomPoints (n):
    list = []
    while n !=0:
        list.append([random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)])
        n = n-1
    return list

def circle ():
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x,y)
    F = X**2 + Y**2 - 1.0
    ax.contour(X,Y,F,[0])

  
def InCircle (x,y):
    d = math.sqrt(x**2 + y**2)
    if d <= 1:
        return 1
    else:
        return 0

# qty = int(input()) for other variants
qty = 100000
time_start = time.clock()

points = randomPoints(qty)
x = [x for x, y in points]
y = [y for x, y in points]
n = 0
for i in range(qty):
  n = n + InCircle(x[i],y[i])
pi = (n/qty)*4
time_elapsed2 = (time.clock() - time_start)

print('After Stochastic:', pi)

fig = plt.figure()
ax = fig.gca()
ax.set_aspect(1)
plt.xlim(-1.25,1.25)
plt.ylim(-1.25,1.25)
ax.grid(linestyle='--')

circle()
ax.scatter(x, y, s=2)
plt.show()

print('Time consuming: ')
print('Trapezoidal method: ', time_elapsed1)
print('Stochastic: ', time_elapsed2)