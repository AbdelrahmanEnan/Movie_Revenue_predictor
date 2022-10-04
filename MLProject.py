#!/usr/bin/env python
# coding: utf-8

# 

# # ML Project (Movie revenue prediction model)

# Problem: For given dataset which includes wide informations of a movie we have to build a model which will preciselly predict the overall worldwide revenue for a movie. 

# ## Defining libraries and Importing data

# In[1]:


import pandas as pd 

import copy

import statistics

import math

import matplotlib.pyplot as plt

import numpy as np

import collections

import json 

from scipy.stats import pearsonr

import ast

#import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
pd.set_option('max_columns', None)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)

import warnings
warnings.filterwarnings('ignore')

movie_dataset = pd.read_csv('C:/Users/Abdo/Desktop/data/py/new.csv')


# ## Analyzing dataset

# In[ ]:


print(movie_dataset.info())
movie_dataset.head()


# In[ ]:


movie_dataset.isnull().sum().sort_values(ascending=False)


# In[2]:


movie_dataset = movie_dataset[movie_dataset.budget.notnull()]
movie_dataset = movie_dataset[movie_dataset.genres.notnull()]
movie_dataset = movie_dataset[movie_dataset.original_language.notnull()]
movie_dataset = movie_dataset[movie_dataset.popularity.notnull()]
movie_dataset = movie_dataset[movie_dataset.production_companies.notnull()]
movie_dataset = movie_dataset[movie_dataset.production_countries.notnull()]
movie_dataset = movie_dataset[movie_dataset.release_date.notnull()]
movie_dataset = movie_dataset[movie_dataset.revenue.notnull()]
movie_dataset = movie_dataset[movie_dataset.runtime.notnull()]
movie_dataset = movie_dataset[movie_dataset.title.notnull()]
movie_dataset = movie_dataset[movie_dataset.vote_count.notnull()]
movie_dataset = movie_dataset[movie_dataset.vote_average.notnull()]
movie_dataset = movie_dataset[movie_dataset.cast.notnull()]
movie_dataset = movie_dataset[movie_dataset.crew.notnull()]
movie_dataset = movie_dataset[movie_dataset.keywords.notnull()]


movie_dataset = movie_dataset[movie_dataset.revenue != 0]
movie_dataset = movie_dataset[movie_dataset.vote_average != 0]
movie_dataset = movie_dataset[movie_dataset.vote_count != 0]
movie_dataset = movie_dataset[movie_dataset.runtime != 0]
movie_dataset = movie_dataset[movie_dataset.budget != '0']
movie_dataset = movie_dataset[movie_dataset.genres != "[]"]
movie_dataset = movie_dataset[movie_dataset.production_companies != "[]"]
movie_dataset = movie_dataset[movie_dataset.production_countries != "[]"]
movie_dataset = movie_dataset[movie_dataset.keywords != "[]"]
movie_dataset = movie_dataset[movie_dataset.cast != "[]"]
movie_dataset = movie_dataset[movie_dataset.crew != "[]"]
print(len(movie_dataset))


# In[ ]:


movie_dataset.columns


# In[ ]:


movie_dataset.isnull().sum().sort_values(ascending=False)


# ## Exploring  variables

# In[3]:


movie_dataset = movie_dataset.reset_index(drop=True)
movie_dataset.reset_index(drop=True, inplace=True)


# In[4]:


# We can clearly see that there are string values which are storend in a json/dictionray like
# datastructure which we will transfer from string to dictionary

movie_dataset[['budget']] = movie_dataset[['budget']].apply(pd.to_numeric) 
movie_dataset[['popularity']] = movie_dataset[['popularity']].apply(pd.to_numeric) 

text_cols = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'keywords', 'cast', 'crew']

def text_to_dict(df):
    for col in text_cols:
        df[col] = df[col].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
    return df

movie_dataset = text_to_dict(movie_dataset)
movie_dataset.info()


# ### Belongs to collection - ostaje

# Belongs_to_collection variables is of type string but i already changed those variables into list datatype ,it represents movies which belongs to a movie franchise , i will mark those movies and that which do not belong to a franchise and analyze the diference between them.Because there are a lot of missing data in this column we can just do it that way.

# In[5]:


part_of_collection = movie_dataset['belongs_to_collection'].apply(lambda x: 1 if len(x) > 0 else 0)


# In[6]:


#part_of_collection = part_of_collection.astype('category')
movie_dataset.belongs_to_collection = part_of_collection


# **Conclusion** : We will add the part of collection column and erease the homepage one

# ### Keywords - neostaju

# In[7]:


allkeywords=[]

def getkey(row):
    x = []
    for i in row:
        allkeywords.append(i['name'])
        x.append(i['name'])
    return x

keywords = movie_dataset['keywords'].apply(lambda x: getkey(x))
keywords


# In[8]:


a = movie_dataset['keywords'].apply(lambda x: len(x) if x != {} else 0)
print(max(a))
# That means that there are movies with 91 keywords


# In[9]:


set1 = set(allkeywords);


# In[10]:


len(set1)   # we have 10569 different Keywords and that just in the training 
            # if we encoded that it would be 10569 new rows


# In[11]:


num = movie_dataset['keywords'].apply(lambda x: len(x) if x != {} else 0)


# In[12]:


movie_dataset.keywords = keywords
movie_dataset['keywords_num'] = num
movie_dataset.keywords_num


# **Conclusion**: since the number of keywords in a movie is not an indicator of the revenue made we could try to use the keywords but made just the top 100

# ### homepage - ostaje

# In[13]:


# Because we have a lot of missing data mostly because not all movies have a homepage and
# thats why i will make a new variable which defines does a movie contains/has a homepage

contain_homepage = movie_dataset['homepage'].isnull().apply(lambda x: 0 if x else 1 )


# In[ ]:


contain_homepage.head(5)


# In[ ]:


sum(contain_homepage) # 1784 have a hompage


# In[14]:


movie_dataset.homepage = contain_homepage


# ### Genres - ostaje

# First ill analyze the colums with the number of genres in it , then ill extract the names and store it in a column

# In[ ]:


movie_dataset.genres[1]


# In[15]:


allgenres =[]

def getgenres(row):
    x = []
    for i in row:
        allgenres.append(i['name'])
        x.append(i['name'])    
    return x

        
genr = movie_dataset['genres'].apply(lambda x: getgenres(x))
gen_num = movie_dataset.genres.apply(lambda x: len(x))


# In[16]:


setgen = set(allgenres)
print(len(setgen))  # We have 20 different genres
setgen


# In[ ]:


genres_count = collections.Counter([i for j in genr for i in j]).most_common()
fig = plt.figure(figsize=(8, 5))
sns.barplot([val[1] for val in genres_count],[val[0] for val in genres_count])
plt.xlabel('Count')
plt.title('Top 20 Genre Count')
plt.show()


# In[ ]:


plt.scatter(gen_num, movie_dataset['revenue'])
plt.show()


# In[ ]:


genr.head()


# In[17]:


movie_dataset.genres = genr
movie_dataset['generes_num'] = gen_num


# **Conclusion**: The column with number of genres is usless and we will not use it , encióding and adding the genres is the best way to use this variable

# ### Crew - ostaje
# 

# In[18]:


def getcrewnum(row):
    return len(row)

crewnumber = movie_dataset['crew'].apply(lambda x: getcrewnum(x))
movie_dataset['crewnumber']=crewnumber


# Next i will take the directors name and make a category from it

# In[19]:


def getdir(row):
    for i in row:
        if i['job'] == 'Director':
            return i['name']

directors = movie_dataset['crew'].apply(lambda x: getdir(x))


# In[ ]:


directors.head(10)


# In[20]:


setd = set(directors);
len(setd) # We have 1858 different directors


# In[21]:


movie_dataset.crew = directors


# Using the crew number is a good indicator for the revenue and can be used , using all directors and encoding it would be to many rows ,maybe ill just use the top 20 directors.

# ### Cast - ostaje

# In[22]:


def getcastnum(row):
    return len(row)

castnumber = movie_dataset['cast'].apply(lambda x: getcastnum(x))


# In[23]:


def getactorsformovie(row):
    x = []
    for i in row:
        if i['order'] in (0,1,2):
            x.append(i['name'])
    return x

castpeople =[]

for row in movie_dataset['cast']:
    for single in row:
        if single['order'] in (0,1,2):
            castpeople.append(single['name'])
        
castnames = movie_dataset['cast'].apply(lambda x: getactorsformovie(x))


# In[ ]:


len(castnames)


# In[24]:


setact = set(castpeople)  # number of names in my new column 


# In[25]:


len(setact) # number of different names


# In[26]:


movie_dataset.cast = castnames
movie_dataset['cast_number'] = castnumber
movie_dataset.columns


# ### Popularity - ostaje

# odradi missing , zeros , hist , boxplot , imali outliners ,mean ,median ,std dev There are no missing data and no zeros in this column

# ### Release date - ostaju

# In[27]:


movie_dataset.release_date = pd.to_datetime(movie_dataset.release_date,  errors='coerce')


# In[28]:


release_year = pd.DatetimeIndex(movie_dataset['release_date']).year
release_month = pd.DatetimeIndex(movie_dataset['release_date']).month
release_dow = pd.DatetimeIndex(movie_dataset['release_date']).dayofweek

movie_dataset = movie_dataset.drop(columns=['release_date'])

movie_dataset['release_dow']= release_dow
movie_dataset['release_year'] = release_year
movie_dataset['release_month'] = release_month


# In[29]:


days = pd.crosstab(index=movie_dataset["release_dow"], columns="count") 
months = pd.crosstab(index=movie_dataset["release_month"], columns="count") 
years = pd.crosstab(index=movie_dataset["release_year"], columns="count") 

days ,months ,years


# In[30]:


movie_dataset.release_dow = movie_dataset.release_dow.astype('category')
movie_dataset.release_month = movie_dataset.release_month.astype('category')


# We have three new columns but we will use only the year because the other two are uniformal

# ### Runtime - ostaje

# ### Budget - ostaje

# ### Production companies - ostaje

# In[ ]:


movie_dataset['production_companies'][0][0]['name']


# In[31]:


def getprodcomnum(row):
    return len(row)

prod_com_number = movie_dataset['production_companies'].apply(lambda x: getprodcomnum(x))


# In[32]:


com = []
def getcomp(row):
    x = []
    for i in row:
        com.append(i['name'])
        x.append(i['name'])
    return x
  
companies = movie_dataset['production_companies'].apply(lambda x: getcomp(x))


# In[ ]:


companies_count = collections.Counter([i for j in companies for i in j]).most_common(20)
fig = plt.figure(figsize=(8, 5))
sns.barplot([val[1] for val in companies_count],[val[0] for val in companies_count])
plt.xlabel('Count')
plt.title('Top 20 Production Company Count')
plt.show()


# In[34]:


setcom = set(com)


# In[35]:


len(setcom)


# In[ ]:


companies.head()


# In[33]:


movie_dataset.production_companies = companies
movie_dataset['companies_num'] = prod_com_number


# ### Production countries - ostaje

# In[ ]:


movie_dataset['production_countries'][0]


# In[36]:


def getprodcouum(row):
    return len(row)

prod_cou_number = movie_dataset['production_countries'].apply(lambda x: getprodcouum(x))


# In[ ]:


print('Median : ' + str(prod_cou_number.median())) , prod_cou_number.describe()


# In[ ]:


plt.scatter(prod_cou_number, movie_dataset['revenue'])
plt.show()


# In[ ]:


pearsonr(prod_cou_number , movie_dataset['revenue'])


# In[37]:


cou = []
def getnat(row):
    x = []
    for i in row:
        cou.append(i['iso_3166_1'])
        x.append(i['iso_3166_1'])
    return x
  
nations = movie_dataset.production_countries.apply(lambda x: getnat(x))


# In[ ]:


countries_count = collections.Counter([i for j in nations for i in j]).most_common(20)
fig = plt.figure(figsize=(8, 5))
sns.barplot([val[1] for val in countries_count],[val[0] for val in countries_count])
plt.xlabel('Count')
plt.title('Top 20 Production Country Count')
plt.show()


# In[39]:


setnat = set(cou)
len(setnat)


# In[38]:


movie_dataset.production_countries = nations
movie_dataset['countries_num'] = prod_cou_number


# ### Original or Spoken language - org ostaje

# For this variable i decided to take the original language as a factor and a new colums which represent 

# In[40]:


movie_dataset['original_language'] = movie_dataset['original_language'].astype('category')


# In[ ]:


movie_dataset.original_language.mode()


# In[ ]:


movie_dataset.original_language.describe()


# In[ ]:


plt.hist(movie_dataset.original_language)
plt.show()


# In[ ]:


movie_dataset.head(5)


# In[41]:


num_spoken = movie_dataset.spoken_languages.apply(lambda x : len(x) if len(x) > 0 else 0) 


# In[42]:


movie_dataset['lang_num'] = num_spoken


# In[ ]:


plt.scatter(num_spoken , movie_dataset.revenue)
plt.show()


# In[ ]:


pearsonr(num_spoken , movie_dataset.revenue) # spoken language in a movie dosnt effect the revenue at all


# We will use categorical original language , spoken we will not use since it is not important 

# ###Votes - ostaje

# In[ ]:


movie_dataset.vote_average.head() 


# In[ ]:


movie_dataset.vote_average.describe()


# In[ ]:


plt.boxplot(movie_dataset.vote_average)
plt.show()


# In[ ]:


plt.scatter(movie_dataset.vote_average , movie_dataset.revenue)
plt.show()


# In[ ]:


pearsonr(movie_dataset.vote_average,movie_dataset.revenue)


# In[ ]:


movie_dataset.vote_count.head()


# In[ ]:


movie_dataset.vote_count.describe()


# In[ ]:


plt.boxplot(movie_dataset.vote_count)
plt.show()


# In[ ]:


plt.scatter(movie_dataset.vote_count , movie_dataset.revenue)
plt.show()


# In[ ]:


pearsonr(movie_dataset.vote_count,movie_dataset.revenue)


# ###Revenue - target

# In[ ]:


movie_dataset.revenue.head(10)


# In[ ]:


movie_dataset.revenue.describe()


# In[ ]:


plt.boxplot(movie_dataset.revenue )
plt.show()


# ## Dropping unwanted variables :

# ### Because of all values are unique: 

# In[43]:


movie_dataset= movie_dataset.drop(columns = ['id' , 'imdb_id' ])


# ### Because we made new variables and dont need them

# In[44]:


movie_dataset = movie_dataset.drop(columns = ['poster_path','overview',
                  'tagline','original_title','title','status', 'adult' ,  'video'  ])


# ### Because they are useless

# In[45]:


movie_dataset = movie_dataset.drop(columns = ['homepage', 'spoken_languages' , 'keywords' ])


# ## Preparing variables for the model

# In[46]:


movie_dataset.columns


# 1. Belongs - ok
# 2. budget - scale
# 3. genres - ok
# 4. org lan - ok 
# 5. prod com - ok
# 6. prod nat - ok
# 7. runtime - scale 
# 8. votes - scale 
# 9. cast - ok
# 10. crew - ok
# 11. releases - ok
# 12. cast number - scale
# 13. release year - scale

# ### Encoding categorical data

# In[47]:


org_data = copy.copy(movie_dataset)


# In[ ]:


movie_dataset = copy.copy(org_data)


# Genre 
# 

# In[48]:


movie_dataset = movie_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(movie_dataset.pop('genres')),
                index=movie_dataset.index,
                columns=mlb.classes_))


# Production countries

# In[49]:


movie_dataset = movie_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(movie_dataset.pop('production_countries')),
                index=movie_dataset.index,
                columns=mlb.classes_))


# Production countries

# In[50]:


movie_dataset = movie_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(movie_dataset.pop('production_companies')),
                index=movie_dataset.index,
                columns=mlb.classes_))


# Cast

# In[52]:


movie_dataset.cast[435][1] = 'Walter Elias Disney' 


# In[57]:


movie_dataset.cast[561][1] = 'Walter Elias Disney' 
movie_dataset.cast[4709][2] = 'Don Blutch'


# In[58]:


index = 0
for i in movie_dataset.cast :
    for single in i:
        if single == 'Coco':
            print(index)
        if single == 'Don Bluth':
            print(index)
        if single == 'Walt Disney':
            print(index)
    index = index +1


# In[59]:


movie_dataset = movie_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(movie_dataset.pop('cast')),
                index=movie_dataset.index,
                columns=mlb.classes_))


# Crew 

# In[60]:


directors = pd.get_dummies(directors, sparse=True)


# In[61]:


dir_col = directors.columns
data_col =   movie_dataset.columns


# In[63]:


cols = []
index = 0
for d in dir_col:
    for m in data_col:
        if d == m :
            cols.append(d)
    index = index +1


# In[64]:


for i in directors.columns:
    if i in cols:
        directors.rename(columns = {i: i + '_dir' }, inplace = True) 


# In[65]:


movie_dataset = movie_dataset.join(directors)


# Original language

# In[66]:


org_lan = pd.get_dummies(movie_dataset.original_language, sparse=True)


# In[67]:


movie_dataset = movie_dataset.join(org_lan)


# Release day of the week 
# 

# In[68]:


release_dow_dummy = pd.get_dummies(movie_dataset.release_dow, sparse=True)


# In[69]:


release_dow_dummy.columns


# In[70]:


release_dow_dummy.rename(columns = {0:'Monday'}, inplace = True) 
release_dow_dummy.rename(columns = {1:'Tuesday'}, inplace = True) 
release_dow_dummy.rename(columns = {2:'Wednesday' }, inplace = True) 
release_dow_dummy.rename(columns = {3: 'Thursday' }, inplace = True) 
release_dow_dummy.rename(columns = {4:'Friday' }, inplace = True) 
release_dow_dummy.rename(columns = {5:'Saturday'}, inplace = True) 
release_dow_dummy.rename(columns = {6:'Sunday'}, inplace = True) 


# In[71]:


release_dow_dummy.columns


# In[72]:


movie_dataset = movie_dataset.join(release_dow_dummy)


# Release month

# In[73]:


release_month_dummy = pd.get_dummies(movie_dataset.release_month, sparse=True)


# In[74]:


movie_dataset = movie_dataset.join(release_month_dummy)


# In[75]:


datatype = movie_dataset.dtypes
useless_columns = datatype[(datatype == 'object') | (datatype == 'category')].index.tolist()
useless_columns


# In[76]:


movie_dataset= movie_dataset.drop(columns = useless_columns)


# In[77]:


datatype = movie_dataset.dtypes
useless_columns = datatype[(datatype == 'object') | (datatype == 'category')].index.tolist()
useless_columns


# In[78]:


movie_dataset.info()


# ### Scaling numberical data 

# ## Model training

# In[80]:


copy_of_data = copy.copy(movie_dataset)
copy_of_data = copy_of_data.sample(frac=1).reset_index(drop=True)
Y = copy_of_data.revenue
X = copy_of_data.drop(columns= ['revenue']) 

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.3 , random_state = 0)


# In[ ]:


#movie_dataset = read_csv('C:/Users/Abdo/Desktop/data/py/new.csv',index=False)


# In[ ]:


movie_dataset=pd.read_csv('new1.csv')
movie_dataset = pd.read_csv("esea_master_dmg_demos.part1.csv")
#movie_dataset.to_csv('C:/Users/Abdo/Downloads/news.csv',index=False)


# In[ ]:





# Linear regression

# In[81]:


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, Y_train)


# In[82]:


predicted = linear_regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(Y_test , predicted)


# In[83]:


Y_test = Y_test.values
print(np.concatenate((predicted.reshape(len(predicted),1), Y_test.reshape(len(Y_test),1)),1))


# Random Forest Regression

# In[84]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 100 , random_state= 0 )
regressor.fit(X_train,Y_train)


# In[85]:


predicted = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(Y_test , predicted)


# In[86]:


Y_test = Y_test.values
print(np.concatenate((predicted.reshape(len(predicted),1), Y_test.reshape(len(Y_test),1)),1))


# ANN

# In[1]:


import keras

from sklearn.neural_network import MLPRegressor
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(100, input_dim=12896, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, epochs=1000, verbose=0)


# In[92]:


sudo pip install keras
pip install keras
import keras


# In[ ]:


accuracy = model.evaluate(X_train, Y_train)
print('Accuracy: %.2f' % (accuracy*100))

predicted = model.predict(X_test)
r2_score(Y_test , predicted)


# In[ ]:


predicted = model.predict(X_test)
from sklearn.metrics import r2_score
r2_score(Y_test , predicted)


# In[ ]:


from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()
X_train = standardScalerX.fit_transform(X_train)
X_test = standardScalerX.fit_transform(X_test)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

