import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def degree(G, node): # True for directional graph
    return (len(G.in_edges(node))+len(G.out_edges(node)))
def in_degree(G,node):
    return (len(G.in_edges(node)))
def out_degree(G,node):
    return (len(G.out_edges(node)))


def dencity(G):
    return G.number_of_edges()/(G.number_of_nodes()*(G.number_of_nodes()-1))

def closeness_centrality(G, u):
    dist = 0
    for v in list(G.nodes):
        if v != u:
            dist += nx.shortest_path_length(G, source=u, target=v)
    return (G.number_of_nodes()-1)/(dist)

def average_shortest_path_length(G, weight=None):
    dist = 0
    for s in list(G.nodes):
        for t in list(G.nodes):
            dist += nx.shortest_path_length(G, source=s, target=t, weight=weight)
    
    return dist/(G.number_of_nodes()*(G.number_of_nodes()-1))

def diameter(G):
    longest = 0
    for s in list(G.nodes):
        for t in list(G.nodes):
            if longest < nx.shortest_path_length(G, source=s, target=t):
                longest = nx.shortest_path_length(G, source=s, target=t)
    return longest

def betwenness (G, v, weight=None):
    shortest = 0
    pathThrough = 0
    for s in list(G.nodes):
        for t in list(G.nodes): 
            if nx.shortest_path(G, source=s, target=t, weight=weight):
                shortest += 1
                if v in nx.shortest_path(G, source=s, target=t, weight=weight):
                    pathThrough += 1
    return pathThrough/shortest

data = pd.read_csv('airports.csv')
G = nx.from_pandas_edgelist(data, source='Origin', target='Dest',edge_attr=True, create_using=nx.DiGraph)
nx.draw_networkx(G, with_labels=True)

print('CRP->BOI with respect to the distance\n',nx.shortest_path(G, source='CRP', target='BOI',weight='Distance'),'\n')
print('BOI->CRP with respect to the distance\n',nx.shortest_path(G, source='BOI', target='CRP',weight='Distance'),'\n')

print('CRP->BOI with respect to the time\n',nx.shortest_path(G, source='CRP', target='BOI',weight='AirTime'),'\n')
print('BOI->CRP with respect to the time\n',nx.shortest_path(G, source='BOI', target='CRP',weight='AirTime'),'\n')

print('Degree connectivity for CRP',G.degree('CRP'),'where inflow centrality:', G.in_degree('CRP'), 'and outflow:', G.out_degree('CRP'))
print('My Degree connectivity for CRP:', degree(G,'CRP'),',inflow :', in_degree(G, 'CRP'),', outflow:', out_degree(G,'CRP'),'\n')
print('Degree connectivity of BOI',G.degree('BOI'),'where inflow centrality:', G.in_degree('BOI'), 'and outflow:', G.out_degree('BOI'))
print('My Degree connectivity for BOI:', degree(G,'BOI'),',inflow :', in_degree(G, 'BOI'),', outflow:', out_degree(G,'BOI'),'\n')

print('Closeness centrality of CRP:', nx.closeness_centrality(G, u='CRP'),', my Closeness centrality:', closeness_centrality(G,'CRP'))
print('Closeness centrality of BOI:', nx.closeness_centrality(G, u='BOI'),', my Closeness centrality:', closeness_centrality(G,'BOI'),'\n')

print('Betweenness centrality pf CRP by Distance:', nx.betweenness_centrality(G, weight='Distance')['CRP'])
print('Betweenness centrality pf CRP by AirTime:', nx.betweenness_centrality(G, weight='AirTime')['CRP'])
print('Betweenness centrality pf BOI by Distance:', nx.betweenness_centrality(G, weight='Distance')['BOI'])
print('Betweenness centrality pf BOI by AirTime:', nx.betweenness_centrality(G, weight='AirTime')['BOI'])
print()

print('Dencity:', nx.density(G), ', my Dencity:', dencity(G))
print('Network diameter:', nx.diameter(G),', my Diameter:', diameter(G))
print('Network average path length with weight Distance:', nx.average_shortest_path_length(G, weight='Distance'),'My average:', average_shortest_path_length(G, weight='Distance'))
plt.show()