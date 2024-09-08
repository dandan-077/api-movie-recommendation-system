import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import streamlit.components.v1 as components

# Load the dataset
try:
    movies = pd.read_csv('tmdb_movies_data.csv')
except FileNotFoundError:
    st.error("The file 'tmdb_movies_data.csv' was not found. Please ensure it is in the correct directory.")
    st.stop()

# Ensure all necessary columns are present
expected_columns = ['overview', 'keywords', 'tagline', 'original_title']
for column in expected_columns:
    if column not in movies.columns:
        st.error(f"Missing expected column: {column}")
        st.stop()

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

# Streamlit UI
st.title('Movie Recommendation System')

# Render the index.html content
with open('index.html', 'r') as file:
    html_content = file.read()

components.html(html_content, height=600)

# Movie title input (assuming the HTML form POSTs to a different URL; adjust as needed)
movie_title = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    if not movie_title:
        st.error("No movie title provided.")
    else:
        recommendations = get_recommendations(movie_title)
        if not recommendations:
            st.error("Movie title not found in dataset.")
        else:
            recommendation_details = []
            for title in recommendations:
                details = fetch_movie_details(title)
                if details:
                    recommendation_details.append(details)
            
            if recommendation_details:
                for movie in recommendation_details:
                    st.subheader(movie['title'])
                    st.write(movie['summary'])
                    if movie['poster']:
                        st.image(movie['poster'])
                    st.write("**Actors:**")
                    for actor in movie['actors']:
                        st.write(f"{actor['name']}")
                        if actor['photo']:
                            st.image(actor['photo'], width=50)
            else:
                st.write("No recommendations found.")
