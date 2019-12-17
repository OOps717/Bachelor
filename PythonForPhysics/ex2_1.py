import scipy
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

x = stats.norm.rvs(loc=1, scale=2, size=100)
y = np.random.uniform(-1, 1, size=1000)
z = np.random.exponential(5, size=1000)

meanN, stdN = 0, 0
for i in range(50):
    meanX, stdX = scipy.mean(x), scipy.nanstd(x)
    meanN += meanX
    stdN += stdX
meanN /= 50
stdN /= 50
print ('Mean and Standard deviation for normal distribution', meanX, stdX)


meanU, stdU = 0, 0
for i in range(50):
    meanY, stdY = scipy.mean(y), scipy.nanstd(y)
    meanU += meanY
    stdU += stdY
meanU /= 50
stdU /= 50
print ('Mean and Standard deviation for uniform distribution', meanY, stdY)

meanE, stdE = 0, 0
for i in range(50):
    meanZ, stdZ = scipy.mean(z), scipy.nanstd(z)
    meanE += meanZ
    stdE += stdZ
meanE /= 50
stdE /= 50
print ('Mean and Standard deviation for exponential distribution', meanZ, stdZ)

shpX = stats.shapiro(x)
if shpX[1] > .05 :
    print ("X distribution is normal, result:", shpX[1])
else:
    print('X distribution is not normal, result:', shpX[1])

shpY = stats.shapiro(y)
if shpY[1] > .05 :
    print ("Y distribution is normal, result:", shpY[1])
else:
    print('Y distribution is not normal, result:', shpY[1])

shpZ = stats.shapiro(z)
if shpZ[1] > .05 :
    print ("Z distribution is normal, result:", shpZ[1])
else:
    print('Z distribution is not normal, result:', shpZ[1])

histX, bin_edgesX = scipy.histogram(x)
histY, bin_edgesY = scipy.histogram(y)
histZ, bin_edgesZ = scipy.histogram(z)
fig, ax = plt.subplots(2,3)
ax[0,0].plot(x)
ax[1,0].bar(bin_edgesX[:-1], histX, width = 1) 
plt.xlim(min(bin_edgesX), max(bin_edgesX))

ax[0,1].plot(y)
ax[1,1].bar(bin_edgesY[:-1], histY, width = 1) 
plt.xlim(min(bin_edgesY), max(bin_edgesY))

ax[0,2].plot(z)
ax[1,2].bar(bin_edgesZ[:-1], histZ, width = 1) 
plt.xlim(min(bin_edgesZ), max(bin_edgesZ))

plt.show() 
print()
data = pd.read_csv('ph.data')
mean, std = scipy.mean(data['pH']), scipy.nanstd(data['pH'])

shp = stats.shapiro(data['pH'])
if shp[1] > .05 :
    print ("pH distribution is normal, result:", shp[1])
else:
    print('pH distribution is not normal, result:', shp[1])

print ('Mean and Standard deviation for pH', mean, std)
plt.hist(data['pH'], bins='auto')
plt.title('pH histogram')
plt.xlabel('Normal')
plt.show()

print()
data = pd.read_csv('cosmic.data')
mean, std = scipy.mean(data['timestamp']), scipy.nanstd(data['timestamp'])
shp = stats.shapiro(data['timestamp'])
if shp[1] > .05 :
    print ("Timestamp distribution is normal, result:", shp[1])
else:
    print('Timestamp distribution is not normal, result:', shp[1])  
print ('Mean and Standard deviation for pH', mean, std)
plt.hist(data['timestamp'], bins='auto')
plt.title('Timestamp histogram')
plt.xlabel('Uniform Distribution')
plt.show()