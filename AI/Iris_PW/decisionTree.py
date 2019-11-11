import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, math, copy

class DecisionTree:
    def __init__(self, data, colNames):
        '''
        Taking data and names of columns for initialization
        '''
        self.data = data
        self.colNames = colNames

    def Attributes (self,columns): 
        '''
        Calculating attributes' quantity and getting their names
        columns - columns' quantity
        '''
        attributes = columns - 1
        attNames = []
        for i in range(attributes):
            attNames.append(self.colNames[i])
        return attributes, attNames

    def Sort (self, attribute, df):
        '''
        Sorting data by attribute
        attribute - attribute by which our data should be sorted
        df - data to sort
        '''
        sorted = df.sort_values(self.colNames[attribute], axis = 0,ascending=True, inplace=False, kind='quicksort', na_position='last')
        return sorted

    def SortByAtt (self, attributes, df):
        '''
        Sorting data by attributes and putting it in list
        attributes - list of columns by which our data should be sorted
        df - data to sort
        '''
        sorted = []
        for i in range(attributes): 
            sorted.append(df.sort_values(self.colNames[i], axis = 0,ascending=True, inplace=False, kind='quicksort', na_position='last'))
        return sorted

    def DiffNames (self, namesCol): 
        '''
        Calculating groups quantity
        namesCol - list of names of instances in all rows
        '''
        qty = len(np.unique(namesCol))
        return qty

    def GroupSort (self, attributes, groupQty, group, sorted): 
        '''
        Making list of all groups sorted by all attributes
        attributes - quantity of attributes
        groupQty - quantity of groups
        group - quantity of instances in the group
        sorted - sorted group of instances to divide
        '''
        groupSorted = []
        for j in range(attributes): 
            k = 0
            for i in range(1,groupQty+1):
                groupSorted.append(sorted[j][k:(group*i)]) 
                k = group*i
        return groupSorted

    def GroupSort2 (self, groupQty, group, sorted):
        ''' 
        The same method as the previous but in this case we are considering that groups are not dividing equally
        groupQty - quantity of groups
        group - quantity of instances in the group
        sorted - sorted group of instances to divide
        '''
        groupSorted = []
        k = 0
        n = 0 
        for i in range(0,groupQty):
            n += group[i] 
            groupSorted.append(sorted[k:n])
            k = n
        return groupSorted

    def Dictionary (self):
        '''
        Creating dicitionary for species
        '''
        d = {}
        gr = self.data[self.colNames[self.data.shape[1]-1]]
        gr = np.unique(gr)
        for i in range(len(gr)):
            if gr[i] not in d.keys():
                d[gr[i]] = i

        return d
    
    def QtyOfSpecies (self, group, qty, rows):
        '''
        Calculating quantity of each specie in certain group or in data
        group - list of names in the group
        qty - quntity of the unique instances
        rows - number of instances in the group
        '''
        
        dic = self.Dictionary()
        l = [0]*3
        for i in range(rows):
            for j in range(3):
                if dic[group[i]] == j:
                    l[dic[group[i]]] = l[dic[group[i]]] + 1

        return l

    def SpeciesInGroup (self, groupSorted, groupQty, attributes, columns, group): 
        '''
        Making list of quantities of each species in each group
        groupSorted - listed of groups sorted by certain atrribute
        groupQty - quantity of groups
        attributes - quantity of attributes
        columns - number of columns
        group - quantity of instances in the group
        d - dictionary for the instances' names
        '''        
        sortedNames  = []
        for i in range(groupQty * attributes): # Taking 'name' column from sorted groups of all attributes and changing it to list
            sortedNames.append(groupSorted[i][self.colNames[columns-1]].tolist()) 
        
        speciesInAll = [] 
        for i in range(groupQty * attributes): # Calculating how many of which species in all groups
            speciesInAll.append(self.QtyOfSpecies (sortedNames[i], groupQty, group))
        
        speciesInGroup = []
        for j in range(1,attributes+1): # Dividing previous calculations by attributes
            s = []
            for i in range((j-1)*groupQty,j*groupQty):
                s.append(speciesInAll[i])
            speciesInGroup.append(s)
        return speciesInGroup

    def Entropy (self, l, ran): 
        '''
        Calculating entropy by the formula:
        Entropy = Sum ((quantity_of_certain_specie/total_number)*log2(quantity_of_certain_specie/total_number))
        l - list of the quantities of each instances
        ran - toatal number of the instances
        '''
        entropy = 0
        for k in range(len(l)):
            if l[k] != 0:
                div = l[k]/ran
                entropy = entropy - (div)*math.log2(div)
        return entropy

    def EntropyGr (self, l, ran):
        '''
        Calculating entropy of each group by the formula:
        Entropy (group) = Sum((quantity_of_certain_specie_in_group/group)*log2(quantity_of_certain_specie_in_group/group))
        l - list of the quantities of each instance in the group
        ran - quantity of the instances in the group
        '''
        ent = []
        for k in range(len(l)):
            ent.append(self.Entropy(l[k],ran))
        return ent   

    def DiscPow (self, entropy, entropyGr,speciesQty, group, totalNumber):
        '''
        Calculating Discriminitive power by the formula:
        Disc(attribute) = Total_entropy - Sum((quantity_of_certain_specie_in_group/group)*entropy_of_group)
        entropy - total entropies
        entropyGr - groups' entropy
        speciesQty - quantity if the certain species in the group
        group - quantity of the instances in the group
        totalNumber - total number of the instances in data
        '''
        discPow = 0
        for i in range(speciesQty):
            discPow += group/totalNumber*entropyGr[i] 
        discPow = entropy - discPow
        return discPow

    def AttributesByDisc (self, unsorted, sorted, columns): 
        '''
        Sorting attributes by discriminitive power
        unsorted - list of discriminitive power of groups in unsorted order
        sorted - list of of discriminitive powers of group from the biggest to the smallest
        columns - columns' names
        '''
        sortedAttributes = []
        columnsInSortedOrder = []
        i = 0
        while i != len(sorted):
            for j in range(len(sorted)):
                if unsorted[i] == sorted[j]:
                    sortedAttributes.append(columns[j])
                    columnsInSortedOrder.append(j)
                    i += 1
                if i == len(columns) - 1:
                    break
        return sortedAttributes, columnsInSortedOrder
    
    def DivideGroupOnDett (self, data, groupQty, attributesOrder, group, lvl):
        '''
        Dividing data by groups in order to achieve groups with unique instances
        data - data to check for unicity
        groupQty - quantity of the groups
        attributesOrder - order of the attributes in reference to discriminitive power
        group - list of numbers of the instances in the group to divide our data
        '''
        unique = False
        if unique == False:
            sorted = self.Sort(attributesOrder[0], data) # Sorting by attributes staring with the biggest discriminitive power
            groupSorted = self.GroupSort2(groupQty, group, sorted) # Dividing by group quantity
            # notUn = []
            for i in range(groupQty):
                if len(np.unique(groupSorted[i][self.colNames[len(attributesOrder)]])) > 1:
                    print('\n\n-------------------------------------------------------------------------------------------------------------------')
                    
                    print('\t\t\t\t\t\t  LEVEL:',lvl,'\n')
                    
                    print('\t\t\t\t\t   Sorted by',self.colNames[attributesOrder[0]])
                    print ('\t\t      ', i + 1, 'group is NOT UNIQUE, instaces are',
                    np.unique(groupSorted[i][self.colNames[len(attributesOrder)]]) ,end = ':\n')
                    print ('\n\t\t\t\t\t  |', groupSorted[i][self.colNames[attributesOrder[0]]].min(), '<', self.colNames[attributesOrder[0]], '<',
                    groupSorted[i][self.colNames[attributesOrder[0]]].max(), '|\n')
                    # print(groupSorted[i])
                    print('---------------------------------PROCEEDING TO NEXT LVL AND CHECKING FOR UNICITY-----------------------------------\n\n')
                    
                    grQty = self.DiffNames(groupSorted[i][self.colNames[len(self.colNames) - 1]].tolist())
                    groupPrev = groupSorted[i].shape[0]
                    for j in range(grQty): # to reduce number of instances in each group
                        group[j] = int(groupPrev/grQty)

                    if group[0] * grQty != groupPrev: # in order if the divided group are not consiting by the same number of instances
                        for j in range(groupPrev - group[0] * grQty):
                            group[j] += 1
                    self.Run(groupSorted[i], group, lvl)
                    # notUn.append(groupSorted[i])
                else:
                    print('\n\n--------------------------------------------------------------------------------------------------------------------')
                    
                    print('\t\t\t\t\t\t  LEVEL:',lvl,'\n')
                    print('\t\t\t\t\t   Sorted by',self.colNames[attributesOrder[0]])
                    print ('\t\t\t    ', i + 1, 'group is UNIQUE','all instaces are',
                    np.unique(groupSorted[i][self.colNames[len(attributesOrder)]]) ,end = ':\n')
                    print ('\n\t\t\t\t\t  |', groupSorted[i][self.colNames[attributesOrder[0]]].min(), '<', self.colNames[attributesOrder[0]], '<',
                    groupSorted[i][self.colNames[attributesOrder[0]]].max(), '|\n')
                    # print(groupSorted[i])
                    print('----------------------------------CHECKING FOR UNICITY IN THE UNCHECKED SETS----------------------------------------\n\n')

                    unique = True
            
                
            


    def Run (self, df, gr, lvl):
        '''
        df - our data to check for unicity
        gr - list of groups of instance by which we can divide our in case if the data is not unique
        lvl - current level of decision tree
        '''
        lvl += 1
        print('\n\n')
        rows, columns = df.shape
        att, attNames = self.Attributes(columns)
        nameCol = df[self.colNames[columns - 1]].tolist()
        sortedByAtt = self.SortByAtt(att,df)
        groupQty = self.DiffNames(nameCol)
        group = int(rows/groupQty)
        
        totalSpecies = self.QtyOfSpecies(nameCol, groupQty, rows)
        groupSorted = self.GroupSort(att, groupQty, group, sortedByAtt)
        speciesInGroups = self.SpeciesInGroup(groupSorted, groupQty, att, columns, group) # the number of occurences of each species
        
        
        entropy = self.Entropy(totalSpecies, rows)
        print ('Total entropy = ', entropy)
        discPow, entropyAllGr = [], []
        for i in range(att):
            entGr = self.EntropyGr(speciesInGroups[i], group)
            print ('Entropy for group of attribute', self.colNames[i], entGr)
            entropyAllGr.append(entGr)
            discPowGr = self.DiscPow(entropy, entGr, groupQty, group, rows)
            discPow.append(discPowGr)
        print('\nAttributes\' names:', attNames)
        print('Discriminative power of each attribute:', discPow)
        
        k = copy.deepcopy(discPow)
        k.sort(reverse = True)
        attributesByDisc, columnsOrdered = self.AttributesByDisc(k, discPow, self.colNames)
        print('Attributes sorted by discriminitive power:', attributesByDisc)
        print('Columns index sorted by discriminitive power:', columnsOrdered)
        print()

        self.DivideGroupOnDett(df,groupQty, columnsOrdered, gr, lvl)






# filePath = input() # iris_text.data in our case
filePath = 'iris_text.data'
data1 = pd.read_csv(filePath) # reading file  
if data1.empty: # In case if there is no such file
    print('Given file is empty')
    sys.exit()
col = ['petal width', 'petal length', 'sepal width', 'sepal length', 'names']
data = pd.read_csv(filePath, names=col) # our working data

'''
Or just...

data = pd.read_csv(filePath)
col = list(data.columns)

...in case if names of columns are initialy set in file
'''


decision = DecisionTree(data, col)
group = decision.QtyOfSpecies(data[decision.colNames[len(col)-1]].tolist(), len(np.unique(data[decision.colNames[len(col)-1]].tolist())), data.shape[0])
decision.Run(data, group, 0)
print ('\n\t\t\t\t\t    NO UNCHECKED SETS LEFT\n')