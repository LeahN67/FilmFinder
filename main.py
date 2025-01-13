import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load the NLP model and vectorizer
filename = './models/nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('./models/tranform.pkl', 'rb'))

def create_similarity():
    """
    Create a similarity matrix for movie recommendations.
    """
    data = pd.read_csv('./data/main_data.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    """
    Generate movie recommendations.
    """
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database.'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # Exclude the first result (the requested movie)
        return [data['movie_title'][a] for a, _ in lst]

def get_suggestions():
    """
    Get movie suggestions for autocomplete.
    """
    data = pd.read_csv('./data/main_data.csv')
    return list(data['movie_title'].str.capitalize())

# Flask Routes
@app.route("/")
def home():
    """
    Render the home page.
    """
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions, api_key=TMDB_API_KEY)

@app.route("/similarity", methods=["POST"])
def similarity():
    """
    Generate recommendations via AJAX.
    """
    movie = request.form['name']
    return "---".join(rcmd(movie))

# Streamlit UI
def streamlit_ui():
    """
    Streamlit-based interface for the app.
    """
    st.title("FilmFinder: Movie Recommendation System 🎥")
    
    # Get movie suggestions
    suggestions = get_suggestions()

    # Input for movie title
    movie_name = st.selectbox("Enter a movie title:", suggestions)

    # Button for generating recommendations
    if st.button("Get Recommendations"):
        if movie_name:
            recommendations = rcmd(movie_name)
            if isinstance(recommendations, list):
                st.subheader("Recommended Movies:")
                for movie in recommendations:
                    st.write(f"- {movie}")
            else:
                st.error(recommendations)
        else:
            st.error("Please select a valid movie title.")

# Main Entry
if __name__ == '__main__':
    # Check for Streamlit Cloud or local Flask
    if os.getenv("IS_STREAMLIT") == "1":
        streamlit_ui()
    else:
        app.run(host='0.0.0.0', port=5000)
