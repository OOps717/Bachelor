import pandas as pd
import numpy as np
import sys, math, copy, random
import matplotlib.pyplot as plt

def shuffle (data,classNum=3):
    '''
    Shuffling data and dividing it by test (30% of the data) and train (70% of the data) 
    
    data - our data
    classNum - number classes ( in our case we have 3 classes : Setosa, Versicolor, Virginica)
    '''
    datas = np.zeros(classNum, dtype=object)
    tests = np.zeros(classNum, dtype=object)
    trains = np.zeros(classNum, dtype=object)
    gr = int(data.shape[0]/3)
    j = 0
    for i in range(3):
        datas[i] = data.iloc[j:(gr*(i+1))]
        datas[i] = datas[i].sample(frac=1)
        tests[i] = datas[i].head(int(0.3*gr))
        trains[i] = datas[i].tail(int(0.7*gr))
        j = gr*(i+1)
    df = pd.concat([datas[0],datas[1],datas[2]], axis=0)
    test = pd.concat([tests[0],tests[1],tests[2]], axis=0)
    train = pd.concat([trains[0],trains[1],trains[2]], axis=0)
    return df, test, train

def softmax(X):
    '''
    Softmax activation function

    X - our input train or test set
    '''
    return (np.exp(X.T) / np.sum(np.exp(X), axis=1)).T

def crossEntropy (yHat, y, classNum=3): 
    '''
    Cross-entropy loss function

    yHat - our output set after activation function
    y - our actual output set
    classNum - number of different classes
    '''
    y = y.T
    yHat = yHat.T
    cost = 0.
    batchSize = len(yHat[0])
    for k in range(classNum):
        for c in range(batchSize):
            cost += y[k][c]*np.log(yHat[k][c]) # Finding cost
    return -(1/batchSize)*cost # Normilizing cost

def epochTrain(w,X,Y,lr=0.05):
    '''
    Applying one epoch of training

    w - weights
    X - input train or test set
    Y - actual output set 
    lr - learning rate (by default 0.05)
    '''
    scores = np.dot(X,w) # Computing weighted inputs
    yHat = softmax(scores) # Activating softmax on weghted inputs
    loss = crossEntropy(yHat,Y) # Calculating loss with cross-entropy
    grad = (-1 / len(X)) * np.dot(X.T,(Y - yHat)) # Calculating gradient for finding new weights
    newW = w - lr*grad # Clalculating new weights
    return loss,grad,newW

def oneHotEncoding(Y, classNum=3):
    '''
    Encoding our inputs with one-hot encoding method to matrix of output

    Y - column of data with outputs
    classNum - number of different classes 
    '''
    Y = np.asarray(Y, dtype='int32') # Converting column to an array
    if len(Y) > 1:
        Y = Y.reshape(-1)
    yMatrix = np.zeros((len(Y), classNum))    
    yMatrix[np.arange(len(Y)), Y] = 1
    return yMatrix

def labelBack(X):
    '''
    Labeling our encoded data

    X - encoded set of irises
    '''
    predicted = X.argmax(axis=1) # Array of the places of the maximum number in each raws
    names = np.zeros(len(predicted), dtype=object)
    for (i,name) in enumerate(predicted):
        if name == 0:
            names[i] = 'Iris-Setosa'
        elif name == 1:
            names[i] = 'Iris-Versicolor'
        else:
            names[i] = 'Iris-Virginica'
    return names


def getAccuracy(out,Y):
    '''
    Calculating accuracy of our calculated output set

    out - calculated output set
    Y - actual output set
    '''
    predicted = out.argmax(axis=1)
    right = 0
    for i in range(len(predicted)):
        if Y[i][predicted[i]] == 1: right += 1 # if calculated value is the actual one
    accuracy = right/len(predicted)*100
    return accuracy


            
col = ['petal width', 'petal length', 'sepal width', 'sepal length', 'names']
df = pd.read_csv('iris_num.data', names=col) # Our working data 

classNum = 3
w = np.random.uniform(size=(df.shape[1]-1, classNum))/10 # Initializing matrix of weights 4x3

dfShuffle, test, train = shuffle(df) # Shuffling data
train = train.sample(frac=1) # Shuffling train set
test = test.sample(frac=1) # Shuffling test set


# Dividing our train set and test to input sets(xTrain, xTest) and output sets(yTrain, yTest)
xTrain = np.zeros((int(train.shape[0]), train.shape[1]-1))
for i in range (int(train.shape[0])):
    for j in range (train.shape[1]-1):
        xTrain[i][j] = train.iloc[i,j]

xTest = np.zeros((int(test.shape[0]), test.shape[1]-1))
for i in range (int(test.shape[0])):
    for j in range (test.shape[1]-1):
        xTest[i][j] = test.iloc[i,j]

yTrain = oneHotEncoding(train['names'])
yTest = oneHotEncoding(test['names'])

epochs = 20000
losses = []
for i in range(epochs):
    loss,grad,w = epochTrain(w,xTrain,yTrain)
    losses.append(loss)
print('Final loss',loss)

print ('Actual labels:\n',labelBack(yTest),'\n') # Printing actual labels of the test data
smax = softmax(np.dot(xTest,w))
print('Obtained labels:\n',labelBack(smax),'\n') # Printing obtained labels 
print('ACCURACY:', getAccuracy(smax,yTest),end='%\n') 

plt.xlabel('Epoch')
plt.ylabel('Cross-entropy Loss')
plt.grid(True)
plt.plot(losses)
plt.savefig('loss.png')
plt.show()
