import numpy as np

def split(word): 
    return [char for char in word]  

def decompose (formula):
    d = {}
    divided = split(formula)
    element = ""
    num = 0

    i,k = 0,0
    while i < len(divided):
    
        if i+1 != len(divided):
            if divided[i].isupper() and divided[i+1].islower():
                element = divided[i]+divided[i+1]
                if element not in d.keys(): d[element] = 1
                else: d[element] += 1
                i+=2
            elif divided[i].isdigit():
                if divided[i+1].isdigit():
                    num  = int(divided[i]+divided[i+1]) -1
                    d[element] += int(num)
                    i+=2
                else:
                    d[element] += int(divided[i]) - 1
                    i+=1
            elif divided[i].isupper():
                element = divided[i]
                if element not in d.keys(): d[element] = 1
                else: d[element] += 1
                i+=1
        else: 
            if divided[i].isupper():
                element = divided[i]
                if element not in d.keys(): d[element] = 1
                else: d[element] += 1
                i+=1
            elif divided[i].isdigit():
                num  = int(divided[i]) -1
                d[element] += int(num)
                i+=1
        # print(d)

    return d

print('Ethanol:', decompose("C2H5OH"))
print('Glucose:', decompose("C6H12O6"))
print('Caffeine:', decompose("C8H10N4O13"))
