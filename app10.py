##import streamlit as st
##import pandas as pd
##from sklearn.feature_extraction.text import TfidfVectorizer
##from sklearn.metrics.pairwise import cosine_similarity
##import requests
##
### Load the dataset with caching
##@st.cache_data
##def load_data():
##    return pd.read_csv('tmdb_movies_data.csv')
##
##movies = load_data()
##
### Ensure all necessary columns are present
##expected_columns = ['overview', 'keywords', 'tagline', 'original_title']
##for column in expected_columns:
##    if column not in movies.columns:
##        st.error(f"Missing expected column: {column}")
##
### Fill missing values and create a combined feature
##movies['combined_features'] = (
##    movies['overview'].fillna('') + ' ' +
##    movies['keywords'].fillna('') + ' ' +
##    movies['tagline'].fillna('')
##)
##
### Initialize the TF-IDF Vectorizer
##tfidf_vectorizer = TfidfVectorizer(stop_words='english')
##
### Fit and transform the combined features into TF-IDF vectors
##tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])
##
### Calculate cosine similarity matrix
##cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
##
### TMDb API Key
##TMDB_API_KEY = '9b6db1f6d700909bc132cf509a1534c7'
##
### Function to get movie recommendations based on similarity
##def get_recommendations(movie_title, cosine_sim=cosine_sim):
##    try:
##        idx = movies[movies['original_title'] == movie_title].index[0]
##    except IndexError:
##        return []
##
##    sim_scores = list(enumerate(cosine_sim[idx]))
##    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
##    sim_scores = sim_scores[1:11]  # Exclude the first item (itself)
##    movie_indices = [i[0] for i in sim_scores]
##    return movies['original_title'].iloc[movie_indices].tolist()
##
### Function to fetch movie details from TMDb
##@st.cache_data
##def fetch_movie_details(movie_title):
##    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
##    search_response = requests.get(search_url).json()
##
##    if 'results' in search_response and search_response['results']:
##        movie_id = search_response['results'][0]['id']
##        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
##        details_response = requests.get(details_url).json()
##
##        movie_data = {
##            "title": details_response.get('original_title'),
##            "summary": details_response.get('overview'),
##            "poster": f"https://image.tmdb.org/t/p/w500{details_response.get('poster_path')}" if details_response.get('poster_path') else None,
##            "actors": [
##                {"name": actor['name'], "photo": f"https://image.tmdb.org/t/p/w500{actor['profile_path']}"}
##                for actor in details_response['credits']['cast'][:5] if actor.get('profile_path')
##            ]
##        }
##        return movie_data
##    return None
##
### Streamlit Interface
##st.title('Movie Recommendation System')
##
##movie_names = movies['original_title'].tolist()
##
##selected_movie = st.selectbox('Choose a movie:', movie_names)
##
##if selected_movie:
##    st.write(f"Recommendations for {selected_movie}:")
##    recommendations = get_recommendations(selected_movie)
##    for rec_movie in recommendations:
##        details = fetch_movie_details(rec_movie)
##        if details:
##            st.subheader(details['title'])
##            if details['poster']:
##                st.image(details['poster'], width=150)
##            st.write(details['summary'])
##            st.write("Actors:")
##            for actor in details['actors']:
##                if actor['photo']:
##                    st.image(actor['photo'], width=50)
##                st.write(actor['name'])



import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

@st.cache_data
def load_data():
    try:
        # Replace with actual data loading logic if needed
        data = pd.DataFrame({
            'original_title': ['Movie1', 'Movie2', 'Movie3'],
            'overview': ['A movie about something', 'Another movie', 'Yet another movie'],
            'keywords': ['action', 'drama', 'comedy'],
            'tagline': ['Tagline1', 'Tagline2', 'Tagline3']
        })
        data['combined_features'] = (
            data['overview'].fillna('') + ' ' +
            data['keywords'].fillna('') + ' ' +
            data['tagline'].fillna('')
        )
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def create_tfidf_matrix(data):
    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim
    except Exception as e:
        st.error(f"Error creating TF-IDF matrix: {e}")
        return None

def get_recommendations(movie_title, data, cosine_sim):
    try:
        idx = data[data['original_title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return data['original_title'].iloc[movie_indices].tolist()
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

def main():
    st.title('Movie Recommendation System')
    
    data = load_data()
    cosine_sim = create_tfidf_matrix(data)
    
    if cosine_sim is None:
        st.error("Error creating cosine similarity matrix. Please check the logs.")
        return
    
    movie_title = st.text_input('Enter a movie title')

    if movie_title:
        recommendations = get_recommendations(movie_title, data, cosine_sim)
        if recommendations:
            st.write('Recommendations:')
            for rec in recommendations:
                st.write(rec)
        else:
            st.write('No recommendations found.')

if __name__ == "__main__":
    main()

