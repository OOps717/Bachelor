import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy, math

def createList(data,listStrartEnd):
    """
    Making list of lists of edges

    data - our working data
    listStartEnd - list of columns name ('Origin', 'Destiny')
    """
    elist = np.zeros(data.shape[0], dtype=object)
    for i in range(data.shape[0]):
        node = list()
        for j in listStrartEnd:
            node.append(data[j][i])
        elist[i] = node
    return elist

def addToGraph(G,node1,node2):
    """
    Adding edges to dictionary of our graph

    G - graph
    orig - city of departure
    dest - city of destination
    """
    if node1 not in G:
        G[node1]= []
        G[node1].append(node2)
    else:
        G[node1].append(node2)
    return G

def dfs(visited, path, graph, From, To):
    """
    Depth-First Search for finding the first path to our destinated city

    visited - visited cities
    pathes - our pathes to the
    graph - our graph
    From - start
    To - end
    """
    if From == To: # If we arrived to destination
        visited.append(From)
        k = copy.deepcopy(visited)
        path.append(k)
    else:
        if From not in visited:
            visited.append(From)
            for neighbour in graph[From]:
                if not path:    
                    dfs(visited, path, graph, neighbour, To)


def evalTime(data,path):
    """
    Calculating time spent on the path

    data - our working data
    path - path to the city
    """
    hours = 0
    for j in range(len(path)-1):
        flights, h = 0, 0
        for k in range(data.shape[0]):
            if data['Origin'][k] == path[j] and data['Dest'][k] == path[j+1]: # if there is a match in origin and destination in data
                if math.isnan(data['AirTime'][k]) == False:
                    h += data['AirTime'][k]
                    flights += 1
        if flights != 0: # In case if the we met nan at the beginnig of calculation
            hours += h/flights # Calculation the average of flights between two airports
    return hours

def evalDist (data,path):
    """
    Calculating the distance of the path

    data - our working data
    path - path to the city
    """
    distance = 0
    for j in range(len(path)-1):
        flights, d = 0, 0
        for k in range(data.shape[0]):
            if data['Origin'][k] == path[j] and data['Dest'][k] == path[j+1]:
                if math.isnan(data['Distance'][k]) == False:
                    d += data['Distance'][k]
                    flights += 1
        if flights != 0:
            distance += d/flights
    return distance

    
data = pd.read_csv('airports.csv')
connections = createList(data, ['Origin','Dest'])
G = {}
for x,y in connections: addToGraph(G,x,y)

visited, path = [], []
dfs(visited, path, G, 'AMA', 'BHM') # NOTE! The written airports is just for example

print('Path from AMA to BHM:', path[0])
print('Average of minutes spent:', evalTime(data, path[0]))
print('Average of the distance of the path:', evalDist(data, path[0]))

G = nx.from_pandas_edgelist(data, source='Origin', target='Dest', edge_attr=True)
nx.draw_networkx(G, with_labels=True)
plt.show()