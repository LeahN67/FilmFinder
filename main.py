import os
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# Flask App Setup
app = Flask(__name__, static_folder='static')

# Load Models
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
    Generate movie recommendations based on a given movie title.
    """
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies.'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        return [data['movie_title'][a] for a, _ in lst]

def get_suggestions():
    """
    Get movie suggestions for the autocomplete feature.
    """
    data = pd.read_csv('./data/main_data.csv')
    return list(data['movie_title'].str.capitalize())

# Flask Routes
@app.route("/")
def home():
    """
    Render the Flask home page with movie suggestions.
    """
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions, api_key=TMDB_API_KEY)

@app.route("/similarity", methods=["POST"])
def similarity():
    """
    Flask API endpoint to generate recommendations.
    """
    movie = request.form['name']
    return "---".join(rcmd(movie))

# Streamlit Interface
def streamlit_ui():
    """
    Streamlit-based UI for the app.
    """
    st.title("🎥 FilmFinder: Movie Recommendation System")

    # Get suggestions
    suggestions = get_suggestions()

    # Input movie title
    movie_name = st.selectbox("Select or enter a movie title:", suggestions)

    # Button to generate recommendations
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

# Entry Point
if __name__ == '__main__':
    # Check if running in Streamlit Cloud or locally
    if os.getenv("IS_STREAMLIT") == "1":  # Streamlit Cloud
        streamlit_ui()
    else:  # Local Flask server
        port = int(os.environ.get("PORT", 5000))
        try:
            app.run(host="0.0.0.0", port=port)
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {port} is already in use. Trying another port...")
                app.run(host="0.0.0.0", port=port + 1)
            else:
                raise e
