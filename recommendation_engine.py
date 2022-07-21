#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from wordcloud import WordCloud
import ast


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')


# In[4]:


credits = pd.read_csv('tmdb_5000_credits.csv')


# In[6]:


movies.head(1)


# In[7]:


credits.head(1)


# In[ ]:


# lets merge both the datasets using title feature.


# In[8]:


df = movies.merge(credits, on = 'title')


# In[10]:


df.shape


# In[12]:


df.head(3)


# In[ ]:


# lets keep only the necessary features.
# those are id,title,genres,keywords,overview, cast,crew


# In[14]:


df1 = df[['id','title','genres','keywords','overview','cast','crew']]


# In[15]:


df1.head()


# In[ ]:


# lets pre process the data.
# that means lets clean the data


# In[28]:


# in the genres, keywords features we do have values as an dictionary. so lets get only the values of the each dictionary.
# for that we can create a user defined function to get the desired data 
def values(text):
    l =[]
    for i in ast.literal_eval(text):
        value = i['name']
        l.append(value)
    return l
        


# In[32]:


df1['genres']=df1['genres'].apply(values)


# In[33]:


df1.head()


# In[35]:


# lets do the same for keywords feature also.

df1['keywords']=df1['keywords'].apply(values)


# In[36]:


df1.head()


# In[ ]:


# now lets do the pre processing for the cast and crew features.
# cast column does have values in dictionary format. each dictionary does have many values. so lets extract only the top 3 names out of each dictionary


# In[41]:


def extract(text):
    li =[]
    count = 0
    for i in ast.literal_eval(text):
        if count!= 3:
            li.append(i['name'])  
            count+=1
        else:
            break
            
    return li
    


# In[43]:


df1['cast']=df1['cast'].apply(extract)


# In[44]:


df1.head()


# In[45]:


df1['crew'][0]


# In[46]:


# lets extract only the director value from ech dictionary of the crew feature.
# for that lets create a user defined function to get the desired data.


def extract1(text):
    lis =[]
    for i in ast.literal_eval(text):
        if i['job']== 'Director':
            lis.append(i['name'])
    return lis
            


# In[48]:


df1['crew'] =df1['crew'].apply(extract1)


# In[49]:


df1.head()


# In[52]:


# as we can observe thare are names in the cast feature. which might be same with the names of crew feature.
# in order to avoid the confusion lets replace space in between first name and second name with no space.
df1['cast'] =df1['cast'].apply(lambda x: [i.replace(' ','') for i in x])


# In[56]:


df1['crew'] = df1['crew'].apply(lambda x: [i.replace(' ','') for i in x])


# In[63]:


df1['keywords'] = df1['keywords'].apply(lambda x: [i.replace(' ','') for i in x])


# In[64]:


df1['genres'] = df1['genres'].apply(lambda x: [i.replace(' ','') for i in x])


# In[65]:


df1.head()


# In[58]:


df1['overview'][0]


# In[69]:


# now lets join all the four rows together.
#for that lets create new column
# here is the problem that the overview column should be in the list format so that we can easily add all the lists together
df1['overview'] = df1['overview'].apply(lambda x : x.split())


# In[70]:


df1.head()


# In[71]:


# lets add the all the four columns to create a big paragraph

df1['tags'] = df1['overview'] + df1['genres'] + df1['cast'] + df1['crew'] + df1['keywords']


# In[73]:


df2= df1[['id','title','tags']]


# In[74]:


df2.head()


# In[76]:


# now we have to pre process the data by using nltak and re libraries.
# so that we can have our data in the correct format

# lets import the necessary libraries

import re
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords


# In[83]:


# first of all lets remove the unnecessary words or special charecters since they will not be useful in processing the data.
# however before going ahead for that lets lower all the words
# to lower the feature lets combine the lists into a string

df2['tags'] =df2['tags'].apply(lambda x : ' '.join(x))


# In[84]:


df2.head()


# In[85]:


# lets lower the string

df2['tags'] = df2['tags'].apply(lambda x : x.lower())


# In[86]:


df2.head()


# In[91]:


# now lets use the porter stemmer in order to avoid the similar words
# for that we need to create a object for the porter stemmer

# for that lets create a function
ps = PorterStemmer()

def test(text):
    z =[]
    for i in text.split():
        j = ps.stem(i)
        z.append(j)
    return ' '.join(z)
        
        


# In[93]:


df2['tags'] =df2['tags'].apply(test)


# In[94]:


df2.head()


# In[97]:


# now we need to convert the tags feture into a vectorized form
# for that we can use bag of model, Tfidf, word2vec
# lets use bag of model for our project
from sklearn.feature_extraction.text import CountVectorizer


# In[98]:


# finally we need to get the cosine distance between each sentence
# for that lets import cosinedistance module

from sklearn.metrics.pairwise import cosine_similarity


# In[100]:


# lets create an object for countvectorizer

cv= CountVectorizer(max_features = 5000, stop_words = 'english')


# In[107]:


vectors = cv.fit_transform(df2['tags']).toarray()


# In[108]:


vectors


# In[110]:


# now lets create a cosine similarity matrics

similarity1 = cosine_similarity(vectors)


# In[112]:


similarity1.shape


# In[113]:


# now we got the similarity matrix 
# now, all we have to do is recommond the movies accoding to the input movie
# for that lets create a function to recommond the movies.
# the input we are giving will be parsed into this function


# In[ ]:


# here we have a problem that the similarity values are not in the descending or ascending order.
# in order to get them sorted we have to consider another problem that, if we sort the values they may loose their position. 
# so that the index value will no more remain same with the index of the movie in out =r final dataframe.
# so that we should first enumerate the similarity of that particular column then change the key value to the value and sort the value according to the key value.
# the proble will be resolved


# In[123]:


# lets consider a single similarity column
sorted(list(enumerate(similarity1[0])),reverse = True, key = lambda x:x[1])[1:5]


# In[165]:


# lets create a function

def recommond(movie,df,similarity1):
    index = df[df['title']== movie].index[0]
    distances = similarity1[index]
    similar = sorted(list(enumerate(distances)),reverse = True, key = lambda x:x[1])[1:5]
    return similar


# In[166]:


x = recommond('Avatar',df2, similarity1)


# In[167]:


for i in x:
    print(df2.iloc[i[0]].title)


# In[169]:


df2.to_csv('recommondation1.csv')


# In[170]:


import requests


# In[171]:


l = 'https://api.themoviedb.org/3/movie/63?api_key=397f9dab81d567d3ac51d55756526eae&language=en-US'


# In[195]:


x = requests.get(l)


# In[197]:


y = x.json()


# In[199]:


y['poster_path']


# In[201]:


x1 =requests.get('https://image.tmdb.org/t/p/w500/gt3iyguaCIw8DpQZI1LIN5TohM2.jpg')


# In[206]:


import io


# In[212]:


from PIL import Image
import requests
from io import BytesIO

response = requests.get('https://image.tmdb.org/t/p/w500/gt3iyguaCIw8DpQZI1LIN5TohM2.jpg')


# In[218]:


img = Image.open(BytesIO(response.content))


# In[219]:


q =np.array(img)


# In[222]:


import matplotlib.pyplot as plt
plt.imshow(q)


# In[ ]:





# In[210]:


with open('x.png','wb') as f:
    f.write(x1)


# In[202]:


with open('x.png','wb') as f:
    f.write(x1)


# In[ ]:





# In[181]:


data = x['poster_path']


# In[182]:


data


# In[192]:


x1 = 'https://image.tmdb.org/t/p/w500/'+ x[]


# In[187]:


import cv2
arr = np.array(x)


# In[190]:


arr.shape


# In[ ]:




