import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('cities_in_az.csv')
G = nx.from_pandas_edgelist(data, source='Origin', target='Destiny',edge_attr=True, create_using=nx.DiGraph)
nx.draw_networkx(G, with_labels=True)

print('Baku->Imishli')
print('Path without cosideration of hours')
print(nx.shortest_path(G, source='Baku', target='Imishli')) # withouth weight
print('Path with cosideration of hours')
print(nx.shortest_path(G, source='Baku', target='Imishli',weight='Hours')) # with weight
''' In the first case we don't apply weights, i.e. that all the edges has the default weight which is 1
=>  The shortest path changes '''


print('\nAdding new path Baku->Imishli with hours taken 1.29')
nx.add_path(G, ['Baku', 'Imishli'])
G.edges['Baku', 'Imishli']['Hours'] = 1.29

print('Baku->Imishli:',nx.shortest_path(G, source='Baku', target='Imishli',weight='Hours'))
print('Imishli->Baku:',nx.shortest_path(G, source='Imishli', target='Baku',weight='Hours'))
''' The path from Baku to Imishli is different if we go in reverse direction as we added the short path only Baku->Imishli,
    but not for Imishli->Baku'''

plt.show()