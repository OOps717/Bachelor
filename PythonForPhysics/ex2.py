import numpy as np
import matplotlib.pyplot as plt

def derivSquare(x,delta):
  der = ((x + delta)**2 - (x)**2)/delta
  return der

def error(expected,real):
  return expected-real

x = np.linspace(0,10,101)
y = derivSquare(x,0.01)

#First task
plt.plot(x,y)
plt.xlabel('points')
plt.ylabel('results')
plt.show()

#Second task
realDeriv = x*2
plt.plot(x,error(y,realDeriv))
plt.xlabel('points')
plt.ylabel('error')
plt.show()

#Third task

l = [0.01,0.001,0.0001]
for i in l:
  y = derivSquare(x,i)
  fig,ax=plt.subplots()
  ax.plot(x,error(y,realDeriv))
  ax.set_xlabel("points")
  ax.set_ylabel("error")

plt.show()
