import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

x = stats.norm.rvs(loc=1, scale=2, size=100)
y = np.random.uniform(-1, 1, size=1000)
z = np.random.exponential(5, size=1000)

meanX, stdX = scipy.mean(x), scipy.nanstd(x)
print ('Mean and Standard deviation for x', meanX, stdX)
meanY, stdY = scipy.mean(y), scipy.nanstd(y)
print ('Mean and Standard deviation for y', meanY, stdY)
meanZ, stdZ = scipy.mean(z), scipy.nanstd(z)
print ('Mean and Standard deviation for z', meanZ, stdZ)

shpX = stats.shapiro(x)
if shpX[1] > .7 :
    print ("X distributes normally, result:", shpX[1])
else:
    print('X distributed not normaly, result:', shpX[1])

shpY = stats.shapiro(y)
if shpY[1] > .7 :
    print ("Y distributes normally, result:", shpY[1])
else:
    print('Y distributed not normaly, result:', shpY[1])

shpZ = stats.shapiro(z)
if shpZ[1] > .7 :
    print ("Z distributes normally, result:", shpZ[1])
else:
    print('Z distributed not normaly, result:', shpZ[1])

histX, bin_edgesX = scipy.histogram(x)
histY, bin_edgesY = scipy.histogram(y)
histZ, bin_edgesZ = scipy.histogram(z)
# ax, fig = plt.subplots(1,3)
ax[0].plot(x)
ax[0].bar(bin_edges[:-1], histX, width = 1) 
plt.xlim(min(bin_edges), max(bin_edges)) 
plt.show() 
