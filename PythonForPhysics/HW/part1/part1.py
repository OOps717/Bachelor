import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit 


def position(p, N): 
    '''
    Claculate the final position after N steps knowing the probability to move forward

    p - probability to move forward
    N - number of steps
    '''
    r = np.random.uniform(size=N)       # randomizing probabilities of each step             

    position = 0                        #initializing the position
    steps = np.zeros(N)
    for i in range(N):
        if r[i] <= p:                   # if random probability is less than p then man goes forward
            steps[i] = 1
            position += 1               # 1 meter forward
        else:
            steps[i] = -1
            position -= 1               # 1 meter backward
    return position,steps

def my_std(N):
    '''
    Finding standard deviation for each step in list of steps

    N - list of steps
    '''
    stds = np.zeros(len(N))                             # initializing list standard deviations
    for j,step in enumerate(N):                         
        guess = np.zeros(100)                           # initializng list of guessed final positions
        for i in range(100):
            guess[i], steps = position(0.5,step)        # guessing final positions
        stds[j] = scipy.nanstd(guess)                   # finding standaard deviation of current step
    return stds

def fit_std(x,a,b):
    return a*np.sqrt(x)+b               # best fitting function for standard deviation is square root function

def guess (times, p, N):
    '''
    Guessing the last position of drunk walker, plotting histogram, calculating standard deviation and testing for normality
    to know wheter the distribution is Gaussian or not
    '''

    guess = np.zeros(times)             # list of geussed final positions

    # Calculating
    for i in range(times):
        guess[i], steps = position(p,N) # getting guessed position and each step after n times of the computation
    avg = np.sum(guess)/100             # calculating average of the final positions after n times of the computation
    
    
    # Getting the histogram and plotting it
    hist, bin_edges = scipy.histogram(guess)
    plt.title("Histogram for probability="+str(p)+" and steps="+str(N))
    plt.bar(bin_edges[:-1], hist, width=5)
    plt.show()

    print("Calculations for",times,"times, with probability =",p,"and steps =",N,":")

    print("Final position:", avg)

    # Getting standard deviation
    std = scipy.nanstd(guess)
    print("Standard deviation:", std)

    # Applying Shapiro test to know if it is Gaussian distribution or not
    shp = stats.shapiro(guess)
    if shp[1] > .05 :
        print ("Shapiro test shows that it is Gaussian distribution, result:", shp[1])
    else:
        print ('Shapiro test shows that it is not Gaussian distribution, result:', shp[1])
    print() 

# Doing computations for 20, 50, 100, 200, 1000 steps with probability 0.5
steps = np.array([20,50,100,200,1000])
for (i,step) in enumerate(steps):
    guess(times=100,p=0.5,N=step)

# Getting different standard deviations for different quantity of steps to fit the function of std to achieve more precise fit
st = np.arange(0,1000,10)
stds = my_std(st)

plt.scatter(st,stds,label='standard deviations')   # scattering achieved standard deviations 
popt, pcov = curve_fit(fit_std, st, stds)   # fitting function, where popt is the values a and b for our fitting in formula a*sqrt(N)+b
plt.plot(st, fit_std(st, *popt), 'g-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))  # plotting fitted function of std
plt.xlabel('steps')
plt.ylabel('std')
plt.legend()
plt.show()

steps = np.array([20,50,100,200,1000])
for (i,step) in enumerate(steps):
    guess(times=100,p=0.75,N=step)

