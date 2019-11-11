import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

k1 = 0.01
k2 = 0.001
k3 = 5

e  = 50
s  = 1000
es = p = 0

out = [[e, s, es, p]]

t = 0

while t <= 20000: # 60,000 ms
    t += np.random.randint(1, 10)
    chance = k1 * e * s + k2 * es + k3 * es
    chance = np.random.rand() * chance
    chance -= k1 * e * s
    if chance <= 0 and e > 0 and s > 0:
        e  -= 1
        s  -= 1
        es += 1
    elif chance > 0:
        chance -= k2 * es
        if chance <= 0 and es > 0:
            es -= 1
            e  += 1
            s  += 1
        elif chance >= 0 and es > 0:
            es -= 1
            e  += 1
            p  += 1
    out.append([e, s, es, p])

out = np.array(out)

fig, axs = plt.subplots(4, 1, figsize=(5, 16))
# axs[0].plot(np.convolve(out[:, 0], [1 / 100 for _ in range(100)], mode="valid"))
axs[0].plot(out[:, 0])
axs[0].set_title("E")

# axs[1].plot(np.convolve(out[:, 1], [1 / 100 for _ in range(100)], mode="valid"))
axs[1].plot(out[:, 1])
axs[1].set_title("S")

# axs[2].plot(np.convolve(out[:, 2], [1 / 100 for _ in range(100)], mode="valid"))
axs[2].plot(out[:, 2])
axs[2].set_title("ES")

# axs[3].plot(np.convolve(out[:, 3], [1 / 100 for _ in range(100)], mode="valid"))
axs[3].plot(out[:, 3])
axs[3].set_title("P")

plt.show()

out = np.zeros([20000, 4])
for _ in tqdm(range(500)):
    k1 = 0.01
    k2 = 0.001
    k3 = 5

    e  = 50
    s  = 1000
    es = p = 0

    t = 0

    while t < 20000: # 60,000 ms
        chance = k1 * e *s + k2 * es + k3 * es
        chance = np.random.rand() * chance
        
        chance -= k1 * e * s
        if chance <= 0 and e > 0 and s > 0:
            e  -= 1
            s  -= 1
            es += 1
        elif chance > 0:
            chance -= k2 * es
            if chance <= 0 and es > 0:
                es -= 1
                e  += 1
                s  += 1
            elif chance >= 0 and es > 0:
                es -= 1
                e  += 1
                p  += 1
        out[t] += np.array([e, s, es, p])
        t += np.random.randint(1, 10)
    
out /= 100

fig, axs = plt.subplots(4, 1, figsize=(5, 16))
axs[0].plot(np.convolve(out[:, 0], [1 / 100 for _ in range(100)], mode="valid")[:10000])
# axs[0].plot(out[:, 0])
axs[0].set_title("E")

axs[1].plot(np.convolve(out[:, 1], [1 / 100 for _ in range(100)], mode="valid")[:10000])
# axs[1].plot(out[:, 1])
axs[1].set_title("S")

axs[2].plot(np.convolve(out[:, 2], [1 / 100 for _ in range(100)], mode="valid")[:10000])
# axs[2].plot(out[:, 2])
axs[2].set_title("ES")

axs[3].plot(np.convolve(out[:, 3], [1 / 100 for _ in range(100)], mode="valid")[:10000])
# axs[3].plot(out[:, 3])
axs[3].set_title("P")

plt.show()