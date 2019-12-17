import numpy as np
import pandas as pd

data = pd.read_csv('AminoAcids.csv',sep=',')
print(data[data['Molecular Weigth'] == data['Molecular Weigth'].max()])
print(data[data['Polarization'] == 'positive'])

print('Number by polarization type: ')
df = data.groupby('Polarization').agg('size').sort_values(ascending=False)
print(df)

df = data.groupby('Polarization')['Molecular Weigth'].agg([('Max Molecular Weigth','max'),('#Polar.type','size')]).sort_values(by='#Polar.type')
print(df)
df = data.groupby('Polarization')['Molecular Weigth'].agg([('Mean Molecular Weigth','mean'),('#Polar.type','size'),('Standard Deviation','std')]).sort_values(by='Mean Molecular Weigth')
print(df)

elements = data.loc[:,'Molecular Formula']

for n in range(len(elements)):
    d = exec(elements[n])
    for el,res in d.items():
        data.loc[n,el] = d[el]


print(data)
print('Biggest number of H atoms:')
print((data.loc[:,'H']).max())

data.to_csv('newAminoAcids.csv',index = False)