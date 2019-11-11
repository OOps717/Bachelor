import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate 

t = np.linspace(0, 100, 120000, endpoint=False)

k1 = 0.01
k2 = 0.001
k3 = 5

e  = 50
s  = 1000
es = p = 0

def func(state, t):
    e, s, es, p = state
    return k2 * es + k3 * es - k1 * e * s, \
           k2 * es - k1 * e * s, \
           k1 * e * s - k2 * es - k3 * es, \
           k3 * es

out = integrate.odeint(func, (e, s, es, p), t)

fig, axs = plt.subplots(4, 1, figsize=(5, 16))
axs[0].plot(out[:, 0])
axs[0].set_title("E")

axs[1].plot(out[:, 1])
axs[1].set_title("S")

axs[2].plot(out[:, 2])
axs[2].set_title("ES")

axs[3].plot(out[:, 3])
axs[3].set_title("P")

np.argmax(out[:, 1] < 5)

plt.show()
