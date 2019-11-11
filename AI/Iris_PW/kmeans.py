import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import sys, math, copy, random


class KMean:

    def __init__(self, data, clustersNum):
        '''
        data - our dataframe
        xlustersNum - number of clusters
        '''
        self.data = data
        self.clustersNum = clustersNum
    
    def points(self):
        '''
        Converting our instance to the points of clusters
        '''
        p = np.zeros((self.data.shape[0], self.data.shape[1]))
        for i in range(len(p)):
            p[i] = self.data.iloc[i,:]
        return p

    def randomCentroids(self, points):
        '''
        Making random centroids
        '''
        centroids = np.zeros((self.clustersNum, self.data.shape[1]), dtype=object)
        for i in range(self.clustersNum):
            centroids[i] = points[random.randrange(0,self.data.shape[0])]
        return centroids
    
    def distance (self, points, centroids):
        '''
        Calculating distances between points and centroid with euclidean distance
        points - all the points of the instance
        centroid - certain centroid
        '''
        distance = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0]):
            d = 0
            for j in range(self.data.shape[1]):
                d += math.pow((points[i][j]-centroids[j]),2)
            d = math.sqrt(d)
            distance[i] = d
        return distance
    
    def distances(self, points, centroids):
        '''
        Calculating distances between points and all centroids
        points - all the points of the instances
        centroids - our current centroids
        '''
        distances = np.zeros((self.clustersNum, self.data.shape[0]))
        for i in range(self.clustersNum):
            distances[i] = self.distance(points,centroids[i])
        return distances

    def calcInClust (self, distances):
        '''
        Calculating how many points are in each cluster and making the order of points in the clusters starting from the first cluster
        distances - distances between points and each cluster
        '''
        inCluster = np.zeros(self.clustersNum)
        order = np.zeros(self.data.shape[0])
        l = 0
        for j in range(len(distances)):
            for i in range(self.data.shape[0]):
                less = True
                for k in range(len(distances)):
                    if j != k:
                        if distances[j][i] < distances [k][i]:
                            less = True
                        else:
                            less = False
                    if less == False:
                        break
                if less == True:
                    inCluster[j] += 1
                    order[l] = i
                    l += 1

        return inCluster, order
    
    def pointsInClust (self, inCluster, points, order):
        '''
        Making a list of list of the points in each clusters
        inCluster - list of quantity of points in each cluster
        points - all the points
        order - order of the points in the clusters
        '''
        clusters = np.zeros(self.clustersNum, dtype = object)
        r1 = 0
        r2 = int(inCluster[0])
        for l in range(self.clustersNum):
            cluster = np.zeros((int(inCluster[l]),self.data.shape[1]), dtype = object)
            j = 0
            for i in range(r1,r2):
                cluster[j] = points[int(order[i])]
                j += 1
            clusters[l] = cluster  
            if l+1 == len(inCluster):
                break  
            r1 = r2
            r2 += int(inCluster[l+1])
        return clusters
    
    def newCent (self, clusters):
        '''
        Calculating new center of the new centroids
        clusters - list of points in each cluster
        '''
        centroids = np.zeros(self.clustersNum, dtype=object)
        for i in range(self.clustersNum):
            centroid = np.zeros((1,self.data.shape[1]))
            for k in range(self.data.shape[1]):
                sumAtt = 0
                for j in range(len(clusters[i])):
                    sumAtt += clusters[i][j][k]
                if len(clusters[i]) != 0:
                    sumAtt /= len(clusters[i])
                centroid[0][k] = sumAtt
            centroids[i] = centroid
        return centroids 
    
    def Run (self):
        p = self.points()
        c = self.randomCentroids(p)
        distances = self.distances(p,c)
        inCluster, order = self.calcInClust(distances)
        clusters = self.pointsInClust (inCluster, p, order) 
        print (inCluster)
        k = True
        while k:
            inClusterPrev = inCluster
            cent = self.newCent(clusters)
            c = np.zeros((self.clustersNum, data.shape[1]))
            for i in range(self.clustersNum):
                c[i] = cent[i]

            distances = self.distances(p,c)
            inCluster, order = self.calcInClust(distances)
            clusters = self.pointsInClust (inCluster, p, order) 
            print (inCluster)

            for i in range(3):
                if inCluster[i] != inClusterPrev[i]:
                    k = True
                    break
                else:
                    k = False
        return clusters, order, inCluster

    def plotClust (self, clusters):
        '''
        HARDCODED, JUST FOR 3 CLUSTERS
        Plotting points
        clusters - points in the cluster
        '''
        coordinates = np.zeros((self.clustersNum), dtype = object)
        for i in range (self.clustersNum):
            coordinate = np.zeros((self.data.shape[1], len(clusters[i])))
            for j in range(len(clusters[i])):
                for k in range(self.data.shape[1]):
                    coordinate[k][j] = clusters[i][j][k]

            coordinates[i] = coordinate
        
        x1 = np.zeros(len(clusters[0]))
        y1 = np.zeros(len(clusters[0]))
        z1 = np.zeros(len(clusters[0]))
        w1 = np.zeros(len(clusters[0]))
        for i in range(len(clusters[0])):
            x1[i] = clusters[0][i][0]
            y1[i] = clusters[0][i][1]
            z1[i] = clusters[0][i][2]
            w1[i] = clusters[0][i][3]


        x2 = np.zeros(len(clusters[1]))
        y2 = np.zeros(len(clusters[1]))
        z2 = np.zeros(len(clusters[1]))
        w2 = np.zeros(len(clusters[1]))
        for i in range(len(clusters[1])):
            x2[i] = clusters[1][i][0]
            y2[i] = clusters[1][i][1]
            z2[i] = clusters[1][i][2]
            w2[i] = clusters[1][i][3]

        x3 = np.zeros(len(clusters[2]))
        y3 = np.zeros(len(clusters[2]))
        z3 = np.zeros(len(clusters[2]))
        w3 = np.zeros(len(clusters[2]))
        for i in range(len(clusters[2])):
            x3[i] = clusters[2][i][0]
            y3[i] = clusters[2][i][1]
            z3[i] = clusters[2][i][2]
            w3[i] = clusters[2][i][3]
        
        fig, ax = plt.subplots(1,4)

        ax[0].scatter(x1, y1)
        ax[0].scatter(x2, y2)
        ax[0].scatter(x3, y3)
        ax[0].set(xlabel='petal width', ylabel='petal length')

        ax[1].scatter(x1, z1)
        ax[1].scatter(x2, z2)
        ax[1].scatter(x3, z3)
        ax[1].set(xlabel='petal width', ylabel='sepal width')
        
        ax[2].scatter(x1, w1)
        ax[2].scatter(x2, w2)
        ax[2].scatter(x3, w3)
        ax[2].set(xlabel='petal width', ylabel='sepal length')
         
        ax[3].scatter(y1, z1)
        ax[3].scatter(y2, z2)
        ax[3].scatter(y3, z3)
        ax[3].set(xlabel='petal width', ylabel='sepal width')
        plt.show()
    
    def accuracy (self, inCluster, order):
        '''
        HARDCODED

        Calculating accuracy of the last clusters for the dataframe of the irises
        inCluster - number of points in each cluster
        order - order of the points in the clusters
        '''
        r1 = 0
        r2 = int(inCluster[0])
        species = np.zeros((self.clustersNum,3))
        for j in range(self.clustersNum):
            for i in range(r1,r2):
                if order[i] < 50:
                    species[j][0] += 1
                elif order[i] >= 50 and order[i]<100:
                    species[j][1] += 1
                else:
                    species[j][2] +=1 

            if j+1 == len(inCluster):
                    break  
            r1 = r2
            r2 += int(inCluster[j+1])

        for i in range(self.clustersNum):
            max = species[i][0]
            maxSp = 0
            for j in range(1,3):
                if  max < species[i][j]:
                    max = species[i][j]
                    maxSp = j
            print('Maximum number of specie in cluster',i,'is',max,'of the',maxSp+1,'specie')
            accuracy = max*100/inCluster[i]
            print('Accuracy for cluster',i,'is', accuracy,'%')
    
    def ClustOrder (self, inCluster, order):
        '''
        Giving a cluster number to each instance
        inCluster - number of points in each cluster
        order - order of the points in the clusters
        '''
        inWhich = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0]):
            cl = 0
            n = 0
            for j in range(self.data.shape[0]):
                same = True
                if n > inCluster[cl]:
                    cl += 1
                    n = 0
                n += 1
                for l in range(self.data.shape[1]):
                    if df.iloc[i,l] != df.iloc[int(order[j]),l]:
                        same = False
                if same:
                    inWhich[i] = cl
        return inWhich






        


# filePath = input() # iris_text.data in our case
filePath = 'iris_text.data'
data1 = pd.read_csv(filePath) # reading file  
if data1.empty: # In case if there is no such file
    print('Given file is empty')
    sys.exit()
col = ['petal width', 'petal length', 'sepal width', 'sepal length', 'names']
df = pd.read_csv(filePath, names=col) # our working data
'''
Or just...

data = pd.read_csv(filePath)
col = list(data.columns)

...in case if names of columns are initialy set in file
'''

cols = [col for col in df.columns if col != "names"]
data = df[cols]

instances = KMean(data, 3)
clusters, order, inCluster = instances.Run()
instances.accuracy(inCluster, order)
# instances.plotClust(clusters)

inWhich = instances.ClustOrder(inCluster, order)

df.insert(data.shape[1]+1,'Cluster',inWhich)
df.to_csv('iris_modified.data', header=False, index=False)

            
        