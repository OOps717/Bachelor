import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('local.csv')

plt.plot(df['length'],df['throughput'])
plt.xlabel('length')
plt.ylabel('throughput')

plt.show()