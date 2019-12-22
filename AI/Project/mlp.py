import numpy as np
import pandas as pd
import sys, math, copy, random
import matplotlib.pyplot as plt
import seaborn as sn


def checkIncreasing(x):
    '''
    Function to check wheter the error on the test set increases on average during 10 consecutive training epochs

    x - list of errors
    '''
    if x[len(x)-11] > x[len(x)-1]:
        return False
    else:
        return True

    
def sigmoid(z): 
    '''
    Sigmoid activation function

    z - weighted inputs
    '''
    return  1/(1+ np.exp(-z))                                     

def sigmoid_(z): 
    '''
    Derivative of sigmoid function

    z - outputs of sigmoid function
    '''    
    return np.multiply(z, 1-z)    

def binaryCross(z,y):
    '''
    Cross-entropy loss function for binary classification

    z - predicted output 
    y - actual output
    '''
    loss = 0
    for i in range(len(y)):
        loss += -y[i]*np.log(z[i])-(1-y[i])*np.log(1-z[i])
    return loss/len(z)           

def shuffle_in_unison(a, b):
    '''
    Shuffling two list in the same way

    a,b - lists
    '''
    assert len(a) == len(b)
    # initializing empty shuffled lists
    shuffled_a = np.empty(a.shape, dtype=a.dtype)                          
    shuffled_b = np.empty(b.shape, dtype=b.dtype)

    # new random indexes
    permutation = np.random.permutation(len(a))    

    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b                       

def divideData (data, noDisease, disease, ratio=0.8):
    '''
    Dividing data by xTrain, xTest, yTrain and yTest with the 80% to the train data and 20% to the test data 
    
    data - our current data
    noDisease - quantity of the samples without having disease
    disease - quantity of the samples with disease
    tr - percentage ratio between train set length and test set length
    '''

    # getting numbers for diseased and not diseased people in train set and test set with respect to ratio
    diseaseTr = int(disease*ratio)                                                                
    noDiseaseTr = int((noDisease+disease)*ratio)-diseaseTr                     
    diseaseTest = disease - diseaseTr
    noDiseaseTest = noDisease - noDiseaseTr

    # dividing data by train and test sets
    train = pd.concat([data.iloc[:diseaseTr],data.iloc[disease:(disease+noDiseaseTr)]],ignore_index=True)
    test = pd.concat([data.iloc[diseaseTr:disease],data.iloc[(disease+noDiseaseTr):(disease+noDiseaseTr+noDiseaseTest)]],ignore_index=True)
    
    # normalizing attributes
    xTrain = normalize(train.drop(['target'], axis = 1)).to_numpy()
    xTest = normalize(test.drop(['target'], axis = 1)).to_numpy()

    # gettintg target outputs
    yTrain = train.target.values
    yTest = test.target.values

    # shuffling attributes list and output lists
    xTrain, yTrain = shuffle_in_unison(xTrain, yTrain)
    xTest, yTest = shuffle_in_unison(xTest, yTest)

    return xTrain, xTest, yTrain, yTest

def normalize(x):
    ''' 
    Normalization of the data of attributes
    Formula: (x - min)/(max-min)

    x - list to normilize
    '''
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def getAccuracy(actual,predicted):
    '''
    Calculating accuracy of our calculated output set
    Formula: (TP+TN)/(FP+TN+TP+FN)

    predicted - calculated output set
    actual - actual output set

    Here:
    TP - people who are actually unhealthy and predicted to be unhealthy
    FP - people who are actually healthy and predicted to be unhealthy
    TN - people who are actually healthy and predicted to be healthy
    FN - people who are actually unhealthy and predicted to be healthy
    '''
    TP, FP, TN, FN = 0,0,0,0
    predicted = np.around(predicted)
    for i in range(len(predicted)):
        if actual[i] == 1 and predicted[i] == 1:
            TP += 1
        elif actual[i] == 0 and predicted[i] == 1:
            FP += 1
        elif actual[i] == 0 and predicted[i] == 0:
            TN += 1
        elif actual[i] == 1 and predicted[i] == 0:
            FN += 1

    return (TP+TN)/(TN+FN+TP+FP) *100
	

def getPrecision (actual,predicted):
    '''
    Calculating accuracy of our calculated output set
    Formula: TP/(FP+TP)

    predicted - calculated output set
    actual - actual output set

    Here:
    TP - people who are actually unhealthy and predicted to be unhealthy
    FP - people who are actually healthy and predicted to be unhealthy
    '''
    TP, FP = 0, 0
    predicted = np.around(predicted)
    for i in range(len(predicted)):
        if actual[i] == 1 and predicted[i] == 1:
            TP += 1
        elif actual[i] == 0 and predicted[i] == 1:
            FP += 1

    if TP==0 and FP==0:             # avoid zero division
        return 0
    else:
        return TP/(TP+FP)

def getSensitivity (actual,predicted):
    '''
    Calculating accuracy of our calculated output set
    Formula: TP/(TP+FN)

    predicted - calculated output set
    actual - actual output set

    Here:
    TP - people who are actually unhealthy and predicted to be unhealthy
    FN - people who are actually unhealthy and predicted to be healthy
    '''
    TP, FN= 0, 0
    predicted = np.around(predicted)
    for i in range(len(predicted)):
        if actual[i] == 1 and predicted[i] == 1:
            TP += 1
        elif actual[i] == 1 and predicted[i] == 0:
            FN += 1

    if TP==0 and FN==0:             # avoid zero division
        return 0
    else:
        return TP/(TP+FN)
        

def getSpecificity (actual,predicted):
    '''
    Calculating accuracy of our calculated output set
    Formula: (TP+TN)/(FP+TN+TP+FN)

    predicted - calculated output set
    actual - actual output set

    Here:
    FP - people who are actually healthy and predicted to be unhealthy
    TN - people who are actually healthy and predicted to be healthy
    '''
    TN, FP = 0, 0
    predicted = np.around(predicted)
    for i in range(len(predicted)):
        if actual[i] == 0 and predicted[i] == 0:
            TN += 1
        elif actual[i] == 0 and predicted[i] == 1:
            FP += 1

    if TN==0 and FP==0:             # avoid zero division
        return 0
    else:
        return TN/(TN+FP)

def confusion (actual,predicted):
    '''
    Creating confusion matrix 2x2:

    TP | FP
    --- ---
    FN | TN

    predicted - calculated output set
    actual - actual output set

    Here:
    TP - people who are actually unhealthy and predicted to be unhealthy
    FP - people who are actually healthy and predicted to be unhealthy
    TN - people who are actually healthy and predicted to be healthy
    FN - people who are actually unhealthy and predicted to be healthy
    '''
    TP, FP, TN, FN = 0,0,0,0
    predicted = np.around(predicted)
    for i in range(len(predicted)):
        if actual[i] == 1 and predicted[i] == 1:
            TP += 1
        elif actual[i] == 0 and predicted[i] == 1:
            FP += 1
        elif actual[i] == 0 and predicted[i] == 0:
            TN += 1
        elif actual[i] == 1 and predicted[i] == 0:
            FN += 1
    return np.array(([TP,FP],[FN,TN]))

def train(xTrain, yTrain, xTest, yTest):
    '''
    Training sets

    xTrain, yTrain - training sets of attributes and outputs
    xTest, yTest - test sets of attributes and outputs
    '''
    nInputs= len(xTrain[0])             # perceptrons in the input layer
    nNeurons= 5                         # perceptrons in the hidden layer
    nOutputs = len(yTrain[0])           # perceptrons in the output layer

    # initializing weights and biases for hidden and output layers
    Wh= 2*np.random.random((nInputs, nNeurons))-1                  
    Bh= 2*np.random.random((len(yTrain), nNeurons))-1
    Wz= 2*np.random.random((nNeurons, nOutputs))-1
    Bz= 2*np.random.random((len(yTrain), nOutputs))-1

    lr = 0.01                   # learning rate
    errors =[]                  # list of error values
    accuracy = []               # list of accuracy values
    precision = []              # list of precision values
    specificity = []            # list of specificity values
    sensitivity = []            # list of sensitivity values

    while True:
        H = sigmoid(np.dot(xTrain, Wh)+Bh)                  # hidden layer results
        Z = sigmoid(np.dot(H, Wz)+Bz)                       # output layer results
        E = yTrain-Z                                        # how much we missed (error)

        hTest = sigmoid(np.dot(xTest, Wh))                  # hidden layer results for test set
        zTest = sigmoid(np.dot(hTest, Wz))                  # output layer results for test set
        accuracy.append(getAccuracy(yTest,zTest))           # calculating accuracy and adding to list 
        precision.append(getPrecision(yTest,zTest))         # calculating precision and adding to list 
        specificity.append(getSpecificity(yTest,zTest))     # calculating specificity and adding to list 
        sensitivity.append(getSensitivity(yTest,zTest))     # calculating sensitivity and adding to list 
        error= binaryCross(zTest,yTest)                     # calculating errors by cross-entropy 
        errors.append(error)
    
        dZ = E * sigmoid_(Z)                                # delta Z
        dH = dZ.dot(Wz.T) * sigmoid_(H)                     # delta H

        # updating values
        Wz += lr*H.T.dot(dZ)                         
        Bz += lr*dZ
        Wh += lr*xTrain.T.dot(dH)                       
        Bh += lr*dH
        
        xTrain, yTrain = shuffle_in_unison(xTrain, yTrain)  # shuffling data after one train epoch

        # checking if the values of the error started to increase
        if (len(errors)>10):
            if checkIncreasing(errors): # break in case if our errors started to increase
                break
    return errors, accuracy, precision, sensitivity, specificity, Wh, Wz 


# reading data
df = pd.read_csv("heart_disease_dataset.csv", sep=';')          

# change the categorical values
sex = pd.get_dummies(df['chest_pain_type'], prefix = "chest_pain_type") 
cp = pd.get_dummies(df['chest_pain_type'], prefix = "chest_pain_type")   
fbs = pd.get_dummies(df['fasting_blood_sugar'], prefix = "fasting_blood_sugar")   
re = pd.get_dummies(df['rest_ecg'], prefix = "rest_ecg")
ea = pd.get_dummies(df['exercise_induced_angina'], prefix = "exercise_induced_angina")
nmv = pd.get_dummies(df['num_major_vessels'], prefix = "num_major_vessels")
th = pd.get_dummies(df['thalassemia'], prefix = "thalassemia")
sl = pd.get_dummies(df['st_slope'], prefix = "st_slope")

frames = [df, sex, cp, fbs, re, ea, nmv, sl, th]                   
df = pd.concat(frames, axis = 1)
df = df.drop(columns = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'num_major_vessels', 'thalassemia', 'st_slope'])

# calculating people with disease and without
noDisease = len(df[df.target == 0])
disease = len(df[df.target == 1])

# dividing data to get train and test sets
xTrain, xTest, yTr, yTt = divideData (df, noDisease, disease)

# changing output of the train and test sets to 2x1 matrix (for calculations)
yTrain = np.zeros((len(yTr),1))
yTest = np.zeros((len(yTt),1))

for i in range(len(yTr)):
    yTrain[i] = yTr[i]
for i in range(len(yTt)):
    yTest[i] = yTt[i]


errors =[]                  # list of error values
accuracy = []               # list of accuracy values
precision = []              # list of precision values
specificity = []            # list of specificity values
sensitivity = []            # list of sensitivity values

errors, accuracy, precision, sensitivity, specificity, Wh, Wz = train (xTrain, yTrain, xTest, yTest)


# plotting errors
plt.plot(errors)
plt.show()

fig, ax = plt.subplots(2,2)
# plotting metrics
ax[0][0].plot(accuracy)
ax[0][0].title.set_text('Accuracy')
ax[0][1].plot(precision)
ax[0][1].title.set_text('Precision')
ax[1][0].plot(specificity)
ax[1][0].title.set_text('Specificity')
ax[1][1].plot(sensitivity)
ax[1][1].title.set_text('Sensitivity')

plt.show()

hTest = sigmoid(np.dot(xTest, Wh))
zTest = sigmoid(np.dot(hTest, Wz))
print("Final accuracy:", getAccuracy(yTest,zTest),'%')
print("Final precision:", getPrecision(yTest,zTest))
print("Final specificity:", getSpecificity(yTest,zTest))
print("Final sensitivity:", getSensitivity(yTest,zTest))

print('Confusion matrix:')
cm = confusion(yTest,zTest)
print(cm)

#plotting confusion matrix
classesY = ['TP - '+str(cm[0][0]),'FP - '+str(cm[0][1])]
classesX = ['TN - '+str(cm[1][0]),'FN - '+str(cm[1][1])]
sn.heatmap(cm, annot=True)

plt.show()