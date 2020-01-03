from settings import token
from yandex_translate import YandexTranslate
import vk
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import unidecode, ast, re, random, sys, copy

def draw(G, measures, measure_name):
    """
    Drawing different centralities

    G - working graph
    measures - dictionary with calculated values of the certain centrality for each node
    measure_name - name of the centrality
    """
    fig = plt.figure(figsize=[15, 8])                                                                                                               # initializing figure
    pos = nx.spring_layout(G)                                                                                                                       # getting positions of the node
    nodes = nx.draw_networkx_nodes(G, pos, node_size=400, cmap=plt.cm.viridis, node_color=list(measures.values()), nodelist=measures.keys())        # drawing nodes and coloring according to its values
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))                                                                                  # normilizing colors value                                                                                  
    
    labels = nx.draw_networkx_labels(G, pos, font_size= 6.5, font_color='#c93400')                                                                  # labeling (comment this part to hide labels)
    edges = nx.draw_networkx_edges(G, pos, width=0.3, edge_color='#6bbdcc')                                                                         # drawig edges
    params = {"xtick.color" : "white",
            "ytick.color" : "white"}
    plt.rcParams.update(params)
    plt.title(measure_name, color='#73cdff')
    plt.colorbar(nodes)
    plt.axis('off')
    fig.set_facecolor("#00000F")
    plt.savefig(measure_name+'.png', facecolor=fig.get_facecolor())
    plt.show()

def add_to_list(id, connections, friends, d, max_connections, connection_per_user):
    """
    Adding connections to the list of connections

    id - current id of the user which has connection with his founded friends
    connections - list of connections between users
    friends - founded friends of the user 
    max_connections - maximum number of connections in the list
    connection_per_user - maximum number of connections for each user 
    """
    attributes = ['user_id','id','first_name','last_name','sex','bdate','country','city','universities','personal']       # defining attributes
    users = np.zeros((len(friends['items']), len(attributes)), dtype=object)                                # initializing list of founded friends
    zeros = 0                                                                                               # initializing number of the deleted users
    for i in range(len(friends['items'])):
        if friends['items'][i]['first_name'] == "DELETED":                                                  # checking if the name of the user is DELETED, means this page is permanently deleted
            zeros += 1                                                                                      
            continue
        for j in range(len(attributes)):                                                                    # making connection between user and its friend
            if j == 0:
                users[i][j] = id                                                                            # adding current user's id for as the first argument in the list
            else:
                if attributes[j] in friends['items'][i]:                                                    # adding founded attributes of the user's friend
                    users[i][j] = friends['items'][i][attributes[j]]
    
    users = users.tolist()                                                                                  
    for i in range(len(users)-zeros):                                                                       # adding friends without deleted friends to the list
        if users[i][0] != 0:    
            if users[i] not in connections and len(connections) < max_connections:                          # checking if the current connection not in the list
                connections.append(users[i])
        if connections_per_user != -1:
            if i == connection_per_user and d!=0:                                                           # making restriction where we can add maximum 20 connections from the current user if it is not initial user(from 0 level)
                break

def make_list(id, connections, visited, max_connections, max_depth, connections_per_user, i=0):
    """
    Making list of the connections. Working principe almost the same as in DFS algorithm

    id - current id of the user which has connection with his founded friends
    connections - list of connections between users
    visited - list of visited edges 
    friends - founded friends of the user 
    i - depth of the search
    """
    if len(connections)< max_connections:                                                   
        friends = api.friends.get(user_id=id, fields="id,first_name,last_name,sex,bdate,country,city,universities,personal", order='name') # finding friends of the current user
        add_to_list(id, connections, friends, i, max_connections, connections_per_user)             # adding possible connections to the list of connections
        print('Investigating', id)
        i += 1
        if i != max_depth:                                                                          # till the second level (investigate friends' friend list)
            for l in connections:                                                                   # going deeper untill we reached the maximum depth or maximum number of connections
                if [id,l[1]] not in visited:                                                        # checking if current edge of connection is not visited
                    if [id,l[1]] not in visited:                                                    # checking if the friend's page is not visited
                        try:
                            visited.append([id,l[1]])                                               # note connection as visited one
                            make_list(l[1], connections, visited, max_connections, max_depth, connections_per_user, i)    # looking for the friends of the current user's friend 
                        except:
                            print('Can not access to user\'s page with id', l[1], len(connections)) # in case if we couldn't reach the friend's page (this page is private, this page is temporaly deleted and etc.)
                    else:                                                                           # if the edge of connection is visited
                        print('Connection is already in list', [id,l[1]])

def connect_border(connections, visited):
    """
    Searching for the connections between the users "on the border"
    
    connections - list of the connections
    visited - list of the visited edges of connections 
    """
    attributes = ['user_id','id','first_name','last_name','sex','bdate','country','city','universities','personal']   # list of attributes
    for i in range(len(connections)):                                                                                                           
        if connections[i][1] not in list(map(list, zip(*connections)))[0]:                  # check if the current user is on the border (if the current node is a leaf)
            try:
                print('Investgating boundary user with id', connections[i][1])
                friends = api.friends.get(user_id=connections[i][1],fields="id,first_name,last_name,sex,bdate,country,city,universities,personal", order='name')
                for j in range(len(friends['items'])):
                    if [connections[i][1],friends['items'][j]['id']] not in visited:        # check if the current edge of connection is not visited
                        if friends['items'][j]['id'] in list(map(list, zip(*connections)))[1] or friends['items'][j]['id'] in list(map(list, zip(*connections)))[0]: # check if the current user's Friend ID is anywhere in the connections
                            visited.append([connections[i][1],friends['items'][j]['id']])   # mark this edge of connection as visited
                            user = np.zeros(len(attributes), dtype=object)                  # initialize new connection 
                            for k in range(len(attributes)):
                                if k == 0:
                                    user[k] = connections[i][1]                             # add for the first attribute current id of the checked user 
                                else:
                                    if attributes[k] in friends['items'][k]:                # add his friend founded attributes
                                        user[k] = friends['items'][j][attributes[k]]   
                            user = user.tolist()
                            connections.append(user)                                        # add connection to list
            except:
                print('Can not access to user\'s page with id',connections[i][1])

def remove_emoji(string):
    """
    Removing emoji unicode from the string

    string - string to modify
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def max_items(d_items):
    """
    Getting list of the items with the same maximum value
    
    d_items - dictionary of items with their quantity as values
    """
    itemMaxValue = max(d_items.items(), key=lambda x: x[1])                 # finding maximum quantity value
    items = []
    # Iterate over all the items in dictionary to find keys with max value
    for key, value in d_items.items():
        if value == itemMaxValue[1]:
            items.append(key)
    return items

def max_four_lang(langs):
    """
    Getting first four languages regarding to the number of users that speak on these languages

    langs - dictionary of the languages with the number of their users as values
    """
    l = copy.deepcopy(langs)
    languages = []
    for i in range(4):
        languages.append(max(l, key=l.get))
        l[languages[i]] = 0
    return languages

def get_person():
    """
    Obtaining the average user according to the qualities of the different users in dataframe and making dictionaries of the 
    """
    df = pd.read_csv('users_connections.csv')
    d = df.drop_duplicates('Friend ID').reset_index(drop=True)                              # getting dataframe without repitition of the users in the friend column
    # You can get your own yandex key by passing through https://passport.yandex.com/auth/list?origin=translate&retpath=https%3A%2F%2Ftranslate.yandex.com%2Fdevelopers%2Fkeys&mode=edit
    translate = YandexTranslate('trnsl.1.1.20200101T142004Z.5ca473d122ac01d7.151c6a9995eb8e52c98d75ad3f266addd27137d9')         # initializing translater
    print("Starting analysis...")

    print("Analysis of the user's sex")
    sexes = {}
    for i in range(d.shape[0]):                                                             # getting the dictionary of the users' sexes and ther quantity
        if d['Friend\'s sex'][i] != 0:                                                      # checking if the sex is indicated
            if d['Friend\'s sex'][i] not in sexes:                                          # checking if founded sex is in the dictionary or not
                sexes[d['Friend\'s sex'][i]] = 1                                            # adding to dictionary if not
            else:
                sexes[d['Friend\'s sex'][i]] += 1                                           # incrementing quantity if it is in dictionary
    rand_sex = random.randrange(0, len(max_items(sexes)))                                   # randomizing the gender number as the quantity of the users' sexes can be equal
    if max_items(sexes)[rand_sex] == 1:                                                     # getting one of sexes with the highest quantity and translating it as string
        sex = "Female"
    else:
        sex = "Male"

    print("Analysis of the user's full name")
    first_names, last_names = {}, {}                                    
    for i in range(d.shape[0]):                                                             # getting the dictionary of the users' first and last names and ther quantity with respect to selected sex
        if d['Friend\'s sex'][i] == max_items(sexes)[rand_sex]:                             # checking if the name and surname are appropriate according to the sex (ex. if sex is male => name should be for men)
            name = unidecode.unidecode(d['Friend\'s first name'][i])                        # rewriting the name into english alphabet
            if name not in first_names:                                                     # adding usage quantity if the name is not in the dictionary or incrementing the quantity if it there
                first_names[name] = 1
            else:
                first_names[name] += 1

            surname = unidecode.unidecode(d['Friend\'s last name'][i])                      # the same procedure as for the name
            if surname not in last_names:
                last_names[surname] = 1
            else:
                last_names[surname] += 1 

    rand_name = random.randrange(0, len(max_items(first_names)))                            # randomizing the name number as the quantity of the users' names can be equal                                  
    name = max_items(first_names)[rand_name]                                                # getting one of names with the highest quantity and translating it as string
    rand_surname = random.randrange(0, len(max_items(last_names)))                          # randomizing the surname number as the quantity of the users' surnames can be equal
    surname = max_items(last_names)[rand_surname]                                           # getting one of surnames with the highest quantity and translating it as string

    print("Analysis of the user's birthday")
    birth_day, birth_month, birth_year = {}, {}, {}
    for i in range(d.shape[0]):                                                             # getting the dictionary of the users' day, month and year of birth and their quantities
        if d['Friend\'s birthday'][i] != '0':                                               # checking if the birthday is indicated
            birthday = d['Friend\'s birthday'][i].split('.')                                # spliting indicated birthday date by day, month and year
            for j in range(len(birthday)):
                if j == 0:                                                                  # if the day is indicated
                    if birthday[j] not in birth_day:                                        # checks whether it is in dictionary or not, increments if it is there  
                        birth_day[birthday[j]] = 1
                    else:
                        birth_day[birthday[j]] += 1 
                elif j == 1:                                                                # same as for day
                    if birthday[j] not in birth_month:
                        birth_month[birthday[j]] = 1
                    else:
                        birth_month[birthday[j]] += 1 
                elif j == 2:                                                                # same as for day
                    if birthday[j] not in birth_year:
                        birth_year[birthday[j]] = 1
                    else:
                        birth_year[birthday[j]] += 1 
    rand_day = random.randrange(0, len(max_items(birth_day)))                               # randomizing the number of the day of birth as the quantity of the users' day of birth can be equal                   
    rand_month = random.randrange(0, len(max_items(birth_month)))                           # randomizing the number of the month of birth as the quantity of the users' month of birth can be equal
    rand_year = random.randrange(0, len(max_items(birth_year)))                             # randomizing the number of the year of birth as the quantity of the users' year of birth can be equal

    birthday = max_items(birth_day)[rand_day]+'.'+max_items(birth_month)[rand_month]+'.'+max_items(birth_year)[rand_year]   # getting final date of birth

    print("Analysis of the user's living country and city")                              
    countries= {}
    for i in range(d.shape[0]):                                                             # getting the dictionary of the countries where users live and their quantities
        if d['Country'][i] != '0':                                                          # checking if the country is indicated
            res = ast.literal_eval(d['Country'][i])                                         # checking the string value and rewriting it to the python expression, if it is one of them, in our case it is dictionary
            if 'title' in res:                                                              # checking if the title of the country is indicated
                if res['title'] not in countries:                                           # checking if the title is in the dictionary or not and increments if it is there
                    countries[res['title']] = 1                                             
                else:
                    countries[res['title']] += 1

    rand_country = random.randrange(0, len(max_items(countries)))                           
    country = max_items(countries)[rand_country]                                            # getting random selected country from the list of the countries with the maximum users in it

    cities = {}
    for i in range(d.shape[0]):                                                             # getting the dictionary of the cities where users live with respect of the selected country and their quantities
        if d['Country'][i] != '0':                                                          # checking firstly if the country is indicated
            res = ast.literal_eval(d['Country'][i])                                         
            if 'title' in res:
                if res['title'] == max_items(countries)[rand_country]:                      # checking if the city belongs to the selected country in order to avoid  misleadings
                    if d['City'][i] != '0':                                                 # cheking if the city is indicated
                        res = ast.literal_eval(d['City'][i])                                # rewriting string into dictionary
                        if 'title' in res:                                                  # checking if the title of the city is indicated
                            if res['title'] not in cities:                                  # manipulating with dictionary
                                cities[res['title']] = 1
                            else:
                                cities[res['title']] += 1  

    rand_city = random.randrange(0, len(max_items(cities)))
    city = max_items(cities)[rand_city]                                                     # getting random selected city from the list of the cities with the maximum users in it

    print("Analysis of the user's personal info and translating in english")
    langs, political, religions, inspired_by, people_main, life_main, smoking, alcohol = {}, {}, {}, {}, {}, {}, {}, {}
    # Getting dictionaries of the diffent personal information and their number in users profile 
    for i in range(d.shape[0]):                                                           
        p = d['Friend\'s personal info'][i]
        if p != '0':                                                                        # checking if the personal info is indicated
            res = ast.literal_eval(p)                                                       # rewriting it into dictionary
            if 'langs' in res:                                                              # checking if the languages are indicated
                for j in range(len(res['langs'])):                                          # going throungh all the mentioned languages and manipulating with dictionary
                    if res['langs'][j] not in langs:
                        langs[res['langs'][j]] = 1
                    else:
                        langs[res['langs'][j]] += 1
            if 'political' in res:                                                          # checking if the political interests are indicated
                if res['political'] not in political:                                       # manipulating with dictionary
                    political[res['political']] = 1
                else:
                    political[res['political']] += 1
            if 'religion' in res and res['religion'] != '':                                 # checking if the religion is mentioned and it is not empty string
                try:                                                                        # trying to translate mentioned religion name while finding the original language
                    text = translate.translate(res['religion'], translate.detect(res['religion'])+'-en')['text'][0].title()
                except:                                                                     # if the original language of the writings is not found or any other exception, using default translation
                    text = translate.translate(res['religion'], 'ru-en')['text'][0].title() # default translation could be changed
                if text not in religions:                                                   # manipulating with the dictionary 
                    religions[text] = 1
                else:
                    religions[text] += 1
            if 'inspired_by' in res:                                                        # checking if the inspirations are indicated
                text = remove_emoji(res['inspired_by'])                                     # removing emojis and unneccessary symbols
                words = re.sub('[!@#$.«»()%^&*_=+"<>:;]', '', text).split(", ")             # spliting text by ', ' to get different inspirations
                for w in words:                                                             # checking and manipulating with each phrase 
                    try:                                                                    # trying to translate and capitalising each word
                        w = translate.translate(w,translate.detect(w)+'-en')['text'][0].title()
                    except:
                        w = translate.translate(w,'ru-en')['text'][0].title()
                    if " And " in w:                                                        # if there is And in the phrase then we are spliting again               
                        ph = w.split(" And ")
                        for j in ph:
                            if j not in inspired_by:
                                inspired_by[j] = 1
                            else:
                                inspired_by[j] += 1
                    else:
                        if w not in inspired_by:
                            inspired_by[w] = 1
                        else:
                            inspired_by[w] += 1
            if 'people_main' in res:                                                        # almost the same procedure as for the previous attributes 
                if res['people_main'] != 0:
                    if res['people_main'] not in people_main:
                        people_main[res['people_main']] = 1
                    else:
                        people_main[res['people_main']] += 1
            if 'life_main' in res:
                if res['life_main'] != 0:
                    if res['life_main'] not in life_main:
                        life_main[res['life_main']] = 1
                    else:
                        life_main[res['life_main']] += 1
            if 'smoking' in res:
                if res['smoking'] != 0:
                    if res['smoking'] not in smoking:
                        smoking[res['smoking']] = 1
                    else:
                        smoking[res['smoking']] += 1
            if 'alcohol' in res:
                if res['alcohol'] != 0:
                    if res['alcohol'] not in alcohol:
                        alcohol[res['alcohol']] = 1
                    else:
                        alcohol[res['alcohol']] += 1
    

    languages = max_four_lang(langs)    # getting the first four languages with the highest usage
    
    # Getting personal information with the highest number of appearance in the user's profile
    polit_poss = ['Communistic','Socialistic','Moderate','Liberal','Conservative','Monarchical','Ultraconservative','Indifferent','Libertarian']
    rand_polit = random.randrange(0, len(max_items(political)))
    political_interest = polit_poss[int(max_items(political)[rand_polit])-1]

    rand_rel = random.randrange(0, len(max_items(religions)))
    religion = max_items(religions)[rand_rel]
    
    rand_insp = random.randrange(0, len(max_items(inspired_by)))
    inspiration = max_items(inspired_by)[rand_insp]

    peop_qual = ['Mind and creativity','Kindness and honesty','Health and beauty','Power and wealth','Courage and perseverance','Humor and love of life']    
    rand_peop = random.randrange(0, len(max_items(people_main)))
    main_in_people = peop_qual[int(max_items(people_main)[rand_peop])-1]
    
    life_prior = ['Family and children','Career and money','Entertainment and relaxation','Science and research','Perfecting the world','Self development','Beauty and art','Fame and influence']
    rand_life = random.randrange(0, len(max_items(life_main)))
    main_in_life = life_prior[int(max_items(life_main)[rand_life])-1]
    
    smoke_rel = ['Sharply negative','Negative','Compromise','Neutral','Positive']
    rand_smoke = random.randrange(0, len(max_items(smoking)))
    relation_to_smoking = smoke_rel[int(max_items(smoking)[rand_smoke])-1]
    
    alc_rel = ['Sharply negative','Negative','Compromise','Neutral','Positive']
    rand_alc = random.randrange(0, len(max_items(alcohol)))
    relation_to_alcohol = alc_rel[int(max_items(alcohol)[rand_alc])-1]

    print("Analysis of the user's higher education")
    # Getting information about university
    university_names = {}
    for i in range(d.shape[0]): 
        if d['Universities'][i] != '0':                                                 # checking if the university is mentioned
            resList = ast.literal_eval(d['Universities'][i])                            # changing it into list firstly
            if res != []:                                                               # if the list of the universities is not empty
                for uni in resList:                                             
                    resDict = ast.literal_eval(str(uni))                                # change each mentioned university to dictionary
                    if 'name' in resDict:                                               # cheking if the name of the university is indicated and manipulating with dictionary
                        if resDict['name'] not in university_names:
                            university_names[resDict['name']] = 1
                        else:
                            university_names[resDict['name']] += 1

    rand_uni_name = random.randrange(0, len(max_items(university_names)))
    university_name = max_items(university_names)[rand_uni_name]

    # Finding faculty name according to the selected university
    faculty_names = {}
    for i in range(d.shape[0]):                                                         # almost the same procedure as for the names of the universities
        if d['Universities'][i] != '0':
            resList = ast.literal_eval(d['Universities'][i])
            if res != []: 
                for uni in resList:
                    resDict = ast.literal_eval(str(uni))
                    if 'name' in resDict:
                        if resDict['name'] == university_name:                          # checking for the faculties only in selected university
                            if 'faculty_name' in resDict and resDict['faculty_name'] != '':
                                fac_name = translate.translate(resDict['faculty_name'], translate.detect(resDict['faculty_name'])+'-en')['text'][0]
                                if fac_name not in faculty_names:
                                    faculty_names[fac_name] = 1
                                else:
                                    faculty_names[fac_name] += 1         

    rand_fac_name = random.randrange(0, len(max_items(faculty_names)))
    faculty_name=translate.translate(max_items(faculty_names)[rand_fac_name],translate.detect(max_items(faculty_names)[rand_fac_name])+'-en')['text'][0].title()

    personality = [name,surname,sex,birthday,country,city,languages,political_interest,religion,inspiration,main_in_people,main_in_life,relation_to_smoking,relation_to_alcohol,university_name,faculty_name]
    return personality, first_names,last_names,sexes,countries,cities,langs,political, religions, inspired_by,people_main, life_main, smoking, alcohol, university_names,faculty_names

def bar_plot(dic, title, names=None):
    """
    Plotting horizontal bar according to the dictionary values

    dic - gicing dictionary ro evaluate
    """
    qty_list, names_list, others = [], [], 0
    sum_keys = sum(dic.values())                                                # getting sum of all values in dictionary
    if len(dic) > 5:                                                            # restricting to print only 5 bars if there are more than 5 keys
        max_names = 4
    else:
        max_names = len(dic)-1
    
    for key in dic.keys():                                                      # writing obtained percentages and key names into lists 
        if len(qty_list) <= max_names:                                          # checking if there is still empty place for bar 
            qty_list.append(dic[max(dic, key=dic.get)]*100/sum_keys)            # calculating percantage
            if names == None:                                                   # checking if we have other name list as in case of political interest and etc.
                names_list.append(max(dic, key=dic.get))  
            else:
                names_list.append(names[int(max(dic, key=dic.get))-1])                                         
        else:
            others += dic[max(dic, key=dic.get)]*100/sum_keys                   # calculating and summing the rest results of key
        dic[max(dic, key=dic.get)] = 0
    if len(dic) > 5:                                                            # checking if there are more than 5 keys so we can add "others" bar
        qty_list.append(others)
        names_list.append("others")

    # Plotting        
    params = {"xtick.color" : "black",
            "ytick.color" : "black"}   
    plt.rcParams.update(params)                                                            
    y_pos = np.arange(len(names_list))
    plt.yticks(y_pos, names_list)
    plt.barh(y_pos, qty_list)
    plt.xlabel("Percentage")
    plt.grid(color='gray', linestyle='--', alpha=0.5)
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()


    
# Initializing session and api
session = vk.Session(access_token=token)        
api = vk.API(session, v='5.65')

# Getting the ID value
id = input("Give the initial id to investigate: ") # as an example 53083705 (page of Dmitriy Medvedev)
if int(id) < 0: 
    print("Wrong id, should be positive number")
    sys.exit()
# Getting the values of restrictions
max_connections = int(input("Give the maximum number of the connections to get (0 - to use default value, -1 - to get maximum number of connections) :"))
if max_connections == 0: max_connections = 1500    
elif max_connections == -1: max_connections = np.inf
elif max_connections < -1: 
    print("Wrong number for maximum connections, should be bigger or equal to -1")
    sys.exit()
connections_per_user = int(input("Give the maximum number of the connections that each user can have (0 - to use default value, -1 - to get maximum number of connection per user) :"))
if connections_per_user == 0: connections_per_user = 50 
elif connections_per_user < -1: 
    print("Wrong number for maximum connections per user, should be bigger or equal to -1")
    sys.exit()
max_depth = int(input("Give the maximum depth that could be reached (0 - to use default value, -1 - to get maximum depth) :"))
if max_depth == 0: max_depth = 2
elif max_depth == -1: max_depth = np.inf
elif max_depth < -1: 
    print("Wrong number for maximum depth, should be bigger or equal to 0")
    sys.exit()

# Initializing list of connections and visited connections
connections, visited = [], []
# Filling list of connections
try:
    make_list(id,connections,visited,max_connections,max_depth,connections_per_user)
except:
    print("Couldn't access to the given id")
    sys.exit()
# Connecting nodes on the border
connect_border(connections,visited)


# Making appropriate dictionary
connections = list(map(list, zip(*connections)))                        # changing rows to columns in list of lists
attributes = ['User ID','Friend ID','Friend\'s first name','Friend\'s last name','Friend\'s sex','Friend\'s birthday','Country','City','Universities','Friend\'s personal info']
d = {}
for i,att in enumerate(attributes):
    d[att] = connections[i]

# Creating dataframe
df = pd.DataFrame(d, columns = attributes)
df.to_csv('users_connections.csv')

# Making graph from data
G = nx.from_pandas_edgelist(df, source=attributes[0], target=attributes[1],edge_attr=True, create_using=nx.Graph)

# Drawing the graph
fig = plt.figure(figsize=[20, 10])
nx.draw(G, with_labels=True, node_color='#005770', node_size=500, font_size=6.5, font_color='#73cdff', width=0.3 ,edge_color='#6bbdcc')
fig.set_facecolor("#00000F")
plt.savefig('user_connections.png', facecolor=fig.get_facecolor())
plt.show()

print("\nStarting analysis of the graph and the data")

# Calculating centralities and drawing the graph according to them
betCent = nx.betweenness_centrality(G)
bet_imp = max(betCent, key=betCent.get)
print("\nThe most important user according to betweenness centrality is the user with id",bet_imp)
draw(G,betCent,"Betweenness_Centrality")

closCent = nx.closeness_centrality(G)
clos_imp = max(closCent, key=closCent.get)
print("\nThe most important user according to closeness centrality is the user with id",clos_imp)
draw(G,closCent,"Closeness_Centrality")

degreeCent = nx.degree_centrality(G)
deg_imp = max(degreeCent, key=degreeCent.get)
print("\nThe most important user according to degree centrality is the user with id",deg_imp)
draw(G,degreeCent,"Degree_Centrality")

eigenCent = nx.eigenvector_centrality(G)
eigen_imp = max(eigenCent, key=eigenCent.get)
print("\nThe most important user according to eigenvector centrality is the user with id",eigen_imp)
draw(G,eigenCent,"Eigenvector Centrality")

print("\nFinding the shortest path between random point on the board and centrality nodes and plotting it...")

rand_friend = random.randrange(0, df.shape[0])
important_cent = list(set([clos_imp,eigen_imp,deg_imp,bet_imp]))
while df['Friend ID'][rand_friend] in important_cent or df['Friend ID'][rand_friend] in list(df['User ID']):
    rand_friend = random.randrange(0, df.shape[0])

for cent in important_cent:                                                                                                                             # going through all centralitiy nodes 
    pos = nx.spring_layout(G)                                                                                                                           # get the postions of nodes
    fig = plt.figure(figsize=[15, 8])
    nx.draw(G, pos, with_labels=True, node_color='#005770', node_size=500, font_size=6.5, font_color='#73cdff', width=0.3 ,edge_color='#6bbdcc')        # drawing graph
    print("Finding the shortest path from",df['Friend ID'][rand_friend]," to",cent)
    path = nx.shortest_path(G,source=df['Friend ID'][rand_friend],target=cent)                                                                          # finding shortest path to the centrality node
    print("The shortest path is:",path)
    path_edges_set = set(zip(path,path[1:]))                                                                                                            # create a set of the edges
    nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='r')                                                                                          # coloring edges and the nodes  on the path to red
    nx.draw_networkx_edges(G,pos,edgelist=path_edges_set,edge_color='r',width=7)
    plt.axis('equal')
    fig.set_facecolor("#00000F")
    plt.show()

person, names, surnames, sexes, countries, cities, langs, political, religions, inspired_by, people_main,life_main,smoking,alcohol,universities,faculties = get_person()

# Plotting bars after analyzation of the data
bar_plot(names,"Names")
bar_plot(surnames,"Surnames")
bar_plot(sexes,"Sexes",['Female','Male'])
bar_plot(countries,"Countries")
bar_plot(cities,"Cities")
bar_plot(langs,"Languages")
bar_plot(political,"Political interests",['Communistic','Socialistic','Moderate','Liberal','Conservative','Monarchical','Ultraconservative','Indifferent','Libertarian'])
bar_plot(religions,"Religions")
bar_plot(inspired_by,"Main inspiration")
bar_plot(people_main,"Main in people",['Mind and creativity','Kindness and honesty','Health and beauty','Power and wealth','Courage and perseverance','Humor and love of life'])
bar_plot(life_main,"Main in life",['Family and children','Career and money','Entertainment and relaxation','Science and research','Perfecting the world','Self development','Beauty and art','Fame and influence'])
bar_plot(smoking,"Relation to smoking",['Sharply negative','Negative','Compromise','Neutral','Positive'])
bar_plot(alcohol,"Relation to alcohol",['Sharply negative','Negative','Compromise','Neutral','Positive'])
bar_plot(universities,"universities")
bar_plot(faculties,"Faculties")

# Getting average person according to connections
qualities = ['Name with the highest usage','Surname with the highest usage','Average sex','Average birthday','Average country'
            ,'Average City','First four languages with the highest usage','Most common political views','Most common religion','Most common inspiration','Main in people',
            'Main in life','Average relation to smoking','Average relation to alcohol','Most common university name','Most common faculty name']
print()
for i in range(len(qualities)):
    print(qualities[i],':',person[i])
