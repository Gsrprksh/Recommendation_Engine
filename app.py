from flask import Flask, render_template, url_for, send_file,Response, request, jsonify, redirect
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
#import ssl




app = Flask(__name__)

#def extract(number, empty_list,empty_list2):
    #url = 'https://api.themoviedb.org/3/movie/{}?api_key=397f9dab81d567d3ac51d55756526eae&language=en-US'.format(number)
    #x1 = requests.get(url)
    #x3 = x1.json()
    #empty_list.append(x3)
    #for i in empty_list:
        #x4 =i['poster_path']
        #empty_list2.append(x4)
    #print(empty_list2)



df = pd.read_csv('recommondation1.csv')

cv = CountVectorizer(max_features = 5000, stop_words='english')
scores = cv.fit_transform(df['tags']).toarray()

similarity = cosine_similarity(scores)


@app.route('/',methods = ['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/recommend', methods = ['POST','GET'])
def recommend():
    #ssl._create_default_https_context = ssl._create_unverified_context
    x2 = request.form['Enter_Movie_Name Here']
    ind = df[df['title'] == x2].index[0]
    distance = similarity[ind]
    reco = sorted(list(enumerate(distance)),reverse = True, key = lambda x:x[1])[0:16]
    li =[]
    li1 =[]
    for i in reco:
        movie_id = df.iloc[i[0]]['id']
        
        url = 'https://api.themoviedb.org/3/movie/{}?api_key=397f9dab81d567d3ac51d55756526eae&language=en-US'.format(movie_id)
        x1 = requests.get(url)
        x3 = x1.json()
        li1.append(x3)
        #empty_list =[]
        #empty_list1=[]
        #data1 = extract(movie_id,empty_list,empty_list1)
        
    return render_template('result.html',data = li1)

if __name__ == "__main__":
    app.run(debug = True)