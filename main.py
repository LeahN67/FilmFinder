import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
# Load the NLP model and TFIDF vectorizer from disk
filename = './models/nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('./models/tranform.pkl', 'rb'))

def create_similarity():
    data = pd.read_csv('./data/main_data.csv')
    # Creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # Creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # excluding first item since it is the requested movie itself
        l = [data['movie_title'][a] for a, _ in lst]
        return l

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    data = pd.read_csv('./data/main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__, static_folder='static')

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions, api_key=TMDB_API_KEY)

@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if isinstance(rc, str):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Get movie suggestions for autocomplete
    suggestions = get_suggestions()

    # Convert necessary strings to lists
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    cast_ids = [x.replace("[", "").replace("]", "") for x in cast_ids.split(",")]

    # Clean cast bios
    cast_bios = [bio.replace(r'\n', '\n').replace(r'\"', '\"') for bio in cast_bios]

    # Combine multiple lists as a dictionary for rendering in HTML
    movie_cards = dict(zip(rec_posters, rec_movies))
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # Web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt').read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})

    reviews_list = [reviews.string for reviews in soup_result if reviews.string]
    reviews_status = ['Good' if clf.predict(vectorizer.transform([review])) else 'Bad' for review in reviews_list]
    movie_reviews = dict(zip(reviews_list, reviews_status))

    api_key = TMDB_API_KEY
    
    return render_template('recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
    vote_count=vote_count, release_date=release_date, runtime=runtime, status=status, genres=genres,
    movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details, api_key=api_key)

if __name__ == '__main__':
    app.run(debug=True)
