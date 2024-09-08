from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)

# Load the dataset
try:
    movies = pd.read_csv('tmdb_movies_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("The file 'tmdb_movies_data.csv' was not found. Please ensure it is in the correct directory.")

# Ensure all necessary columns are present
expected_columns = ['overview', 'keywords', 'tagline', 'original_title']
for column in expected_columns:
    if column not in movies.columns:
        raise ValueError(f"Missing expected column: {column}")

# Fill missing values and create a combined feature
movies['combined_features'] = (
    movies['overview'].fillna('') + ' ' +
    movies['keywords'].fillna('') + ' ' +
    movies['tagline'].fillna('')
)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# TMDb API Key
TMDB_API_KEY = '9b6db1f6d700909bc132cf509a1534c7'

# Function to get movie recommendations based on similarity
def get_recommendations(movie_title, cosine_sim=cosine_sim):
    try:
        idx = movies[movies['original_title'] == movie_title].index[0]
    except IndexError:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first item (itself)
    movie_indices = [i[0] for i in sim_scores]
    return movies['original_title'].iloc[movie_indices].tolist()

# Function to fetch movie details from TMDb
def fetch_movie_details(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    search_response = requests.get(search_url).json()

    if search_response['results']:
        movie_id = search_response['results'][0]['id']
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
        details_response = requests.get(details_url).json()

        movie_data = {
            "title": details_response['original_title'],
            "summary": details_response['overview'],
            "poster": f"https://image.tmdb.org/t/p/w500{details_response['poster_path']}" if details_response['poster_path'] else None,
            "actors": [
                {"name": actor['name'], "photo": f"https://image.tmdb.org/t/p/w500{actor['profile_path']}"}
                for actor in details_response['credits']['cast'][:5] if actor['profile_path']
            ]
        }
        return movie_data
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('movie_title', default=None, type=str)
    if not movie_title:
        return jsonify({"error": "No movie title provided."}), 400
    
    recommendations = get_recommendations(movie_title)
    if not recommendations:
        return jsonify({"error": "Movie title not found in dataset."}), 404

    recommendation_details = []
    for title in recommendations:
        details = fetch_movie_details(title)
        if details:
            recommendation_details.append(details)
    
    return jsonify({"recommendations": recommendation_details})

if __name__ == '__main__':
    app.run(debug=True)
