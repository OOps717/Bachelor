import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CheckUp():
    def __init__(self, attribute, value):
        '''
        attribute - attribute with max discriminitive power 
        value - value according to this attribute
        '''
        self.attribute = attribute
        self.value = value
    
    def check(self, sample):
        '''
        Checking if the data is the right one

        sample - sample to check
        '''
        return sample[0, self.attribute] == self.value

class DecisionTreeClassifier():
    
    def __init__(self, data, maxLevel=4, numGroup=4, indexes="init", attributeExtracted=[], groups=[], test=None):
        '''
        data - data to train
        maxLevel - max depth of the tree
        numGroup - max number of the categorical groups
        indexes - indexes of the instances (True/False)
        attributeExtracted - list of the attributes which were already been with max discriminitive power
        groups - list of maximums of each attribute for each category
        test - test for verification of data if it is right or not
        '''
        self.data = data
        self.numGroup = numGroup
        self.maxLevel = maxLevel
        self.attributeExtracted = attributeExtracted
        self.groups = groups
        self.levels = []
        self.test = test
        self.encode()                                                                   # encoding                                                                

        if str(indexes) == "init": self.indexes = np.ones(len(data[0]), dtype=bool)     # initializing list of indexes
        else: self.indexes = indexes.copy()

        if maxLevel > 0: self.levels = self.build()                                     # building the tree
    
    def gettest(self): 
        '''
        Getting test for the data
        '''
        return self.test

    def encode(self, data=[]):
        '''
        Encoding samples while sorting it by categories
        
        data - data to encode
        '''
        if len(data) == 0:                                                      # encoding for the initial data                                               
            if len(self.groups) != 0: return self.data[0]                       # if there are not any sample groups => no need in encoding             
            data = self.data[0].copy()
            groups = np.zeros((self.data[0].shape[1], self.numGroup))           # creating list of groups

            for attribute in range(data.shape[1]):
                data = data[data[:, attribute].argsort()]                       # sorting arguments of attribute in ascending order
                for i in range(self.numGroup):                                  
                    current = int(data[int((i + 1)*data.shape[0]/self.numGroup - 1), attribute])  # starting index of samples of each group
                    groups[attribute, i] = current
            
            df = np.empty(data.shape)                                           # new encoded data
            
            for attribute in range(groups.shape[0]):
                for i in range(groups.shape[1]):
                    # each value of attribute that is less than or equal to the current group value and bigger than the previos one is equal to its index
                    if i == 0: df[:, attribute][self.data[0][:, attribute] <= groups[attribute, i]] = i   
                    else:
                        df[:, attribute][(self.data[0][:, attribute] <= groups[attribute, i]) & (self.data[0][:, attribute] > groups[attribute, i - 1])] = i 
                df[:, attribute][self.data[0][:, attribute] > groups[attribute, -1]] = self.numGroup - 1

            self.groups = groups.copy()
            self.data = (df.copy(), self.data[1])

        else:
            groups = self.groups
            df = np.empty(data.shape)

            for attribute in range(groups.shape[0]):
                for i in range(groups.shape[1]):
                    # each value of attribute that is less than or equal to the current group value and bigger than the previos one is equal to its index
                    if i == 0: df[:, attribute][data[:, attribute] <= groups[attribute, i]] = i
                    else: df[:, attribute][(data[:, attribute] <= groups[attribute, i]) & (data[:, attribute] > groups[attribute, i - 1])] = i
                df[:, attribute][data[:, attribute] > groups[attribute, -1]] = self.numGroup - 1
        return df

    def entropy(self, attribute=None):
        '''
        Finding general entropy and entropy of one attribute 
        '''

        entropy = 0
        uniques, counts = np.unique(self.data[1][self.indexes], return_counts=True)                                 # finding unique target values
        
        if attribute == None:                                                                                       # case of finding general entropy
            for u,c in zip(uniques, counts):      
                div = c/len(self.data[1][self.indexes])
                entropy-= div*np.log2(div)

        else:                                                                                                       # case of finding unique attributes of attribute
            uniqueAtt, countAtt = np.unique(self.data[0][self.indexes, attribute], return_counts=True)             
            for uAtt, cAtt in zip(uniqueAtt, countAtt):
                index = np.logical_and(self.indexes, (self.data[0][:, attribute].reshape(-1) == [uAtt]))            # finding indexes of the current unique value of attribute 
                ent = 0
                for u in uniques:
                    y = len(self.data[1][np.logical_and(index, (self.data[1].reshape(-1) == [u]))])                 # finding quantitiy of the current target value according to the value of an attribute 
                    if y != 0:                                                                                      # to avoid log2(0)
                        div = y / cAtt                                                                              
                        ent += div * np.log2(div) 
                entropy-= (cAtt / len(self.indexes) * ent)
        return entropy

    
    def discPow(self, attribute):
        '''
        Finding discriminitive power of the attribute

        attribute - selected attribute
        '''
        return self.entropy() - self.entropy(attribute)

    def build(self):
        '''
        Building a decision tree by the use of discriminative power
        '''

        levels = []                                                                     # list ov levels of the desicion tree
        maxDisc = -1                                                                    # max discriminitive power 
        attMaxDisc = 0                                                                  # attribute with this discriminitive power 

        for attribute in range(self.data[0].shape[1]):
            if attribute in self.attributeExtracted: continue              
            disc = self.discPow(attribute)                                 
            if disc > maxDisc:                                                          # finding max discriminitive power and its attribute
                maxDisc = disc                                              
                attMaxDisc = attribute                                              

        uniques = np.unique(self.data[0][self.indexes, attMaxDisc])                     # unique values among the values of attribute with max discriminitive power
    
        attributeExtracted = []
        attributeExtracted.append(attMaxDisc)                                           # extracting attribute with max discrimitive power
        
        for u in uniques:
            indexes = np.logical_and(self.indexes, self.data[0][:, attMaxDisc] == u)    # writing new indexes (True - current attribute)
            tests = CheckUp(attMaxDisc, u)                                              # adding new rule
            levels.append(DecisionTreeClassifier(self.data, self.maxLevel - 1, self.numGroup, indexes, attributeExtracted, self.groups, tests))  # building new bracnch
            
        return levels
    

    
    def guess(self, sample):
        '''
        Guessing target value of the sample
        
        sample - current sample to guess
        '''
        if len(self.levels) == 0:                                                               # if it the last level of the tree
            uniques, counts = np.unique(self.data[1][self.indexes], return_counts=True)         # it checks for the unique targets and counts them
            return uniques[np.argmax(counts)]                                                   # returns the target with the max quantity 
        else:
            for b in self.levels:
                if b.gettest().check(sample):                                                   # finding needed branch according to the test's result
                    branch = b
                    break
            return branch.guess(sample)                                                         # proceeding to this branch

    
    def predict(self, test):
        '''
        Predicting targets of the test data using desicion tree

        test - test data
        '''

        results = np.empty((test.shape[0], 1), dtype=self.data[1].dtype)                        # list of predicted targets
        data = test.copy()                          
        data = self.encode(data)                                                                # encoding test data

        for i,sample in enumerate(data): results[i, 0] = self.guess(sample.reshape(1, -1))      # guessing        
        
        return results


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

def normalize(x):
    ''' 
    Normalization of the data of attributes
    Formula: (x - min)/(max-min)

    x - list to normilize
    '''
    return (x - np.min(x)) / (np.max(x) - np.min(x))


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

    xTrain = normalize(train.drop(['target'], axis = 1)).to_numpy()
    xTest = normalize(test.drop(['target'], axis = 1)).to_numpy()

    # gettintg target outputs
    yTrain = train.target.values
    yTest = test.target.values

    # shuffling attributes list and output lists
    xTrain, yTrain = shuffle_in_unison(xTrain, yTrain)
    xTest, yTest = shuffle_in_unison(xTest, yTest)

    return xTrain, xTest, yTrain, yTest

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


df = pd.read_csv("heart_disease_dataset.csv", sep=';')

# Correlation
plt.rc('xtick', labelsize=6)   
plt.rc('ytick', labelsize=8)   
plt.matshow(df.corr())
plt.yticks(np.arange(df.shape[1]), df.columns)
plt.xticks(np.arange(df.shape[1]), df.columns, rotation='vertical')
plt.colorbar()
plt.show()

noDisease = len(df[df.target == 0])
disease = len(df[df.target == 1])

xTrain, xTest, yTr, yTt = divideData (df, noDisease, disease)

yTrain = np.zeros((len(yTr),1))
yTest = np.zeros((len(yTt),1))

for i in range(len(yTr)):
    yTrain[i] = yTr[i]
for i in range(len(yTt)):
    yTest[i] = yTt[i]


dt = DecisionTreeClassifier((xTrain, yTrain), numGroup=4, maxLevel=4)
pred = dt.predict(xTest)

print("Accuracy   :", getAccuracy(pred, yTest))
print("Precision  :", getPrecision(pred, yTest))
print("Sensitivity:", getSensitivity(pred, yTest))
print("Specificity:", getSpecificity(pred, yTest))