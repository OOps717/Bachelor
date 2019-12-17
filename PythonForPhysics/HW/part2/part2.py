import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

def fit_ent(x,a,b):
    return a*np.log(x)+b       # best fitting function for entropy            

def entropyComp (computations=25, p=0.2 ,N=50):
    '''
    Entropy calculation for average probability after several computations 

    computations - number of computations
    p - probability of particle to stay at current position
    N - number of steps of the particle
    '''
    puddle = np.zeros((21,21)) # as the particle can be even on the cell 0 and 20 (board)
    cellsOccupied = 0        

    for i in range(computations): # calculating number of the particle in each cell and total number of particles after computations           
        x,y = move(p=p,N=N)
        for j in range(len(x)):
            puddle[int(x[j])][int(y[j])] += 1
            cellsOccupied += 1
            
    k = 1.38 * (np.power(np.e,-23)) # Boltzman constant 

    prob = np.zeros((21,21)) # probabilities list of particle being in each cell
    sumP = 0
    for i in range(21):
        for j in range(21):
            if puddle[i][j] !=0:                               # to avoid log(0)
                prob[i][j] = puddle[i][j]/cellsOccupied        # average probability for each cell 
                sumP += prob[i][j]*np.log(prob[i][j])/np.log(np.e)
    entropy = -k * sumP 
    return entropy

def getDirection(x,y,p):
    '''
    Getting direction of particle in one step by taking into consideration the probability of the particle to stay ath the same place
    and the directions are equiprobable

    x - current position of the particle on the x axis
    y - current position of the particle on the y axis
    p - probability of the particle not to move
    '''
    r = np.random.uniform()                         # randomizing probability of the particle to move
    if r <= p:
        return 's'
    elif x > 0 and x < 20 and y > 0 and y < 20 :    # if particle in the center
        if r <= 0.25*(1-p)+p:
            return 'l'         
        elif r <= 2*0.25*(1-p)+p:
            return 'u'          
        elif r <= 3*0.25*(1-p)+p:
            return 'r'          
        else:
            return 'd'          
    elif x == 20 and y < 20 and y > 0:              # if particle in the right side
        if r <= 0.33*(1-p)+p:
            return 'u'          
        elif r <= 2*0.33*(1-p)+p:
            return 'r'          
        else:
            return 'd'          
    elif x == 0 and y < 20 and y > 0:               # if particle in the left side
        if r <= 0.33*(1-p)+p:
            return 'u'          
        elif r <= 2*0.33*(1-p)+p:
            return 'l'          
        else:
            return 'd'          
    elif x < 20 and x > 0 and y == 0:               # if particle in the upper side
        if r <= 0.33*(1-p)+p:
            return 'l'          
        elif r <= 2*0.33*(1-p)+p:
            return 'u'          
        else:
            return 'r'          
    elif x < 20 and x > 0 and y == 20:               # if particle in the down side
        if r <= 0.33*(1-p)+p:
            return 'l'          
        elif r <= 2*0.33*(1-p)+p:
            return 'd'          
        else:
            return 'r'                  
    elif x == 0 and y == 0:                         # if particle in the top right part
        if r <= 0.5*(1-p)+p:
            return 'u'         
        else:
            return 'r'                    
    elif x == 20 and y == 0:                        # if particle in the top left part
        if r <= 0.5*(1-p)+p:
            return 'u'          
        else:
            return 'l'          
    elif x == 0 and y == 20:                        # if particle in the down right part
        if r <= 0.5*(1-p)+p:
            return 'd'          
        else:
            return 'r'         
    elif x == 20 and y == 20:                       # if particle in the down left part
        if r <= 0.5*(1-p)+p:
            return 'd'         
        else:
            return 'l'       


def move(p,N):
    '''
    Getting the last position of the particles after N steps and with respect to probability not ot move
    '''
    # Initializing positions of the particle
    x = np.zeros(400)
    y = np.zeros(400)
    
    # Initial positions of particles in center 4 cells
    for i in range(len(x)):
        for i in range(100):
            x[i] = 9
            y[i] = 10
        for i in range(100,200):
            x[i] = 9
            y[i] = 9
        for i in range(200,300):
            x[i] = 10
            y[i] = 10
        for i in range(300,400):
            x[i] = 10
            y[i] = 9

    for n in range(N):
        for i in range(len(x)):
            dir = getDirection(x[i],y[i],p) # getting direction of each particle for each step
            if dir == 'l':
                x[i] -= 1
            if dir == 'u':
                y[i] += 1
            if dir == 'r':
                x[i] += 1
            if dir == 'd':
                y[i] -= 1
            if dir == 's':
                continue
            # in case if particle is on 21 or 0 position , i.e. out of boards
            if x[i] == 21: x[i] = 20
            if x[i] == -1: x[i] = 0
            if y[i] == 21: y[i] = 20
            if y[i] == -1: y[i] = 0
    return x,y


steps = [5,10,20,50]
for i in steps:
    x,y = move(p=0.2,N=i)   # getting final positions of particle in N steps
    plt.scatter(x,y)
    plt.title(str(i)+' steps')
    plt.xlim((0,20))
    plt.ylim((0,20))
    plt.show()

#-------------Calculating and ploting the entropy--------------------------
steps = np.array([5,10,20,50])
entropies = np.zeros(len(steps))
for i,step in enumerate(steps):
    entropies[i] = entropyComp(N=step)                                                    # getting entropies after several steps
print(entropies)

plt.plot(steps,entropies,label='entropy')                                                 # plotting obtained entropies
popt, pcov = curve_fit(fit_ent, steps, entropies)                                         # finding a and b for fitting function 
plt.plot(steps, fit_ent(steps, *popt), 'g-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))  # plotting fitted function of std
plt.xlabel('steps')
plt.ylabel('entropies')
plt.legend()
plt.show()