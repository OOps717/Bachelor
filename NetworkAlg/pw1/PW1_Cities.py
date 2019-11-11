import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

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

def addToGraph(G,orig,dest):
    """
    Adding edges to dictionary of our graph

    G - graph
    orig - city of departure
    dest - city of destination
    """
    if orig not in G:
        G[orig]= []
        G[orig].append(dest)
    else:
        G[orig].append(dest)
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
    if From == To:
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
        for k in range(data.shape[0]):
            if data['Origin'][k] == path[j] and data['Destiny'][k] == path[j+1]:
                hours += data['Hours'][k]
    return hours

    
data = pd.read_csv('cities_in_az.csv')
connections = createList(data, ['Origin','Destiny'])
G = {}
for x,y in connections: addToGraph(G,x,y)

visited, pathes = [], []
dfs(visited, pathes, G, 'Kurdamir', 'Imishli') # NOTE! The written cities is just for example

print('Path from Kurdamir to Imishli:',pathes[0])
hours = evalTime(data, pathes[0])
print('Time spent travelling:',hours)

# G = nx.from_pandas_edgelist(data, source='Origin', target='Destiny', edge_attr=True)
G = nx.DiGraph()
G.add_edges_from(connections)
nx.draw_networkx(G, with_labels=True)
plt.show()