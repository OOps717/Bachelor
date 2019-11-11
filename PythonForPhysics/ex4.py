import numpy as np

a = np.array([[4,-1,-1,-1],[-1,3,0,-1],[-1,0,3,-1],[-1,-1,-1,4]])
b = np.array([15,15,0,0])
x = np.linalg.solve(a, b)   

print(x)

# We can use PySpice as the simulation of electrical circuits, but it was difficult for me to use it gave errors