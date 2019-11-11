from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import matplotlib.pyplot as plt

def lorenz(x, y, z, s=10, r=28, b=3):
    xl = s*(y - x)
    yl = r*x - y - x*z
    zl = x*y - b*z
    return xl, yl, zl


points = 10000
dt = 0.02

xs = np.empty(points + 1)
ys = np.empty(points + 1)
zs = np.empty(points + 1)

xs[0], ys[0], zs[0] = (1., 1., 1.)

for i in range(points):
    xl, yl, zl = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (xl * dt)
    ys[i + 1] = ys[i] + (yl * dt)
    zs[i + 1] = zs[i] + (zl * dt)


fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=0.09)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz")

fig,ax=plt.subplots()
ax.plot(xs,ys,linewidth=0.09)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')

fig,ax=plt.subplots()
ax.plot(ys,zs,linewidth=0.09)
ax.set_xlabel('y axis')
ax.set_ylabel('z axis')

fig,ax=plt.subplots()
ax.plot(xs,zs,linewidth=0.09)
ax.set_xlabel('x axis')
ax.set_ylabel('z axis')

plt.show()