from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from cb_recommender import ContentBasedRecommender
from cf_recommender import CollaborativeRecommender
from sklearn.neighbors import NearestNeighbors
from simulate_mab import mab_on_contentbased, mab_on_collabfilter
from utils import load_movielens_data, pearson_distance
from director_recommender import recommend_by_movie_id, recommend_films_with_actors
from fuzzywuzzy import process

app = Flask(__name__)
CORS(app)

class MovieRecommenderApp:
    def __init__(self, data_dir: str = "./dataset/"):
        self.data_dir = data_dir
        self.df_movies = None
        self.df_ratings = None
        self.df_tags = None
        self.df_with_abstracts = None
        self.content_recommender = None
        self.collaborative_recommender = None
        self.utility_matrix = None
        self.content_initialized = False
        self.collaborative_initialized = False
        self.director_initialized = False
        self.movies_with_abstracts_path = "./dataset/movies_with_abstracts_complete.csv" #! Cambiare path con uno dinamico
        self._load_basic_data()

    def _load_basic_data(self) -> None:
        try:
            self.df_movies, self.df_ratings, self.df_tags = load_movielens_data(self.data_dir)
            self
            self.utility_matrix = self.df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
            self.df_with_abstracts = pd.read_csv(self.movies_with_abstracts_path, dtype={'imdbId': str, 'tmdbId': str})
        except Exception as e:
            print(f"Errore nel caricamento dei dati base: {e}")

    def movie_finder(self, title):
        all_titles = self.df_with_abstracts['title'].tolist()
        closest_match = process.extractOne(title, all_titles)
        return closest_match[0]

    def initialize_content_recommender(self) -> None:
        if self.content_initialized:
            return
        try:
            self.content_recommender = ContentBasedRecommender(self.df_with_abstracts)
            self.content_initialized = True
        except Exception as e:
            print(f"Errore nell'inizializzazione del content recommender: {e}")

    def initialize_collaborative_recommender(self, n_neighbors: int = 10) -> None:
        if self.collaborative_initialized:
            return
        try:
            knn_model_item = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1)
            knn_model_user = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1)
            self.collaborative_recommender = CollaborativeRecommender(knn_model_item, knn_model_user)
            self.collaborative_recommender.fit_item_model(self.utility_matrix, re_fit=True)
            self.collaborative_recommender.fit_user_model(self.utility_matrix, re_fit=True)
            self.collaborative_initialized = True
        except Exception as e:
            print(f"Errore nell'inizializzazione del collaborative recommender: {e}")

    def check_director_recommender(self) -> bool:
        if os.path.exists(self.movies_with_abstracts_path):
            self.df_with_abstracts = pd.read_csv(self.movies_with_abstracts_path)
            self.director_initialized = True
            return True
        else:
            return False

    def search_movie_by_title(self, query: str) -> list:
        if self.df_with_abstracts is None:
            return []
        results = self.df_with_abstracts[self.df_with_abstracts['title'].str.contains(query, case=False, regex=False)]
        movies_list = []
        for _, row in results.iterrows():
            movies_list.append(row)
        return movies_list

    def show_movie_details(self, movie_id: int) -> dict:
        if self.df_with_abstracts is None:
            return {}
        try:
            movie = self.df_with_abstracts[self.df_with_abstracts['movieId'] == movie_id].iloc[0].to_dict()
            if self.df_ratings is not None:
                ratings = self.df_ratings[self.df_ratings['movieId'] == movie_id]
                if len(ratings) > 0:
                    avg_rating = ratings['rating'].mean()
                    num_ratings = len(ratings)
                    movie['avg_rating'] = avg_rating
                    movie['num_ratings'] = num_ratings
            return movie
        except (IndexError, KeyError):
            return {}

    def run_content_recommender(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        if not self.content_initialized:
            self.initialize_content_recommender()
            if not self.content_initialized:
                return pd.DataFrame()
        return self.content_recommender.recommend(movie_title, top_n=top_n)
    
    def run_content_recommender_mab(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        return mab_on_contentbased(movie_title, self.df_ratings, 20)[:top_n]

    def run_collaborative_item_recommender(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return pd.DataFrame()
        return self.collaborative_recommender.get_item_recommendations(movie_id, self.df_movies).head(top_n)
    
    def run_collaborative_item_recommender_mab(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        return mab_on_collabfilter(self.df_ratings, self.df_movies, movie_id, None, N=20)

    def run_collaborative_user_recommender(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return pd.DataFrame()
        return self.collaborative_recommender.get_user_recommendations(user_id, self.utility_matrix, self.df_movies).head(top_n)
    
    def run_collaborative_user_recommender_mab(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        return mab_on_collabfilter(self.df_ratings, self.df_movies, None, user_id, N=20)[:top_n]


    def run_director_recommender_by_movie(self, movie_id: int, max_actors: int = 5) -> str:
        if not self.check_director_recommender():
            return "Director recommender not available."
        return recommend_by_movie_id(self.movies_with_abstracts_path, movie_id, max_actors=max_actors, movie_title_selected=True)

    def run_director_recommender_by_director(self, director: str, max_actors: int = 5) -> str:
        if not self.check_director_recommender():
            return "Director recommender not available."
        return recommend_films_with_actors(director, max_actors=max_actors, movie_title_selected=False)

app_instance = MovieRecommenderApp()

@app.route('/search', methods=['POST'])
def search_movies():
    data = request.get_json()
    query = data.get('query', '')
    results = app_instance.search_movie_by_title(query)
    results = [result.to_dict() for result in results]
    return jsonify(results)

@app.route('/movie/<int:movie_id>', methods=['GET'])
def movie_details(movie_id):
    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts['movieId'] == movie_id]

    if(movie_id <= 0):
        return jsonify({
            "error": True,
            "message": "Movie ID must be a positive integer."
        }), 404
    
    if len(movie) == 0:
        return jsonify({
            "error": True,
            "message": f"Movie with ID {movie_id} not found."
        }), 404
    
    details = app_instance.show_movie_details(movie_id)
    return jsonify(details)

@app.route('/recommend/content', methods=['POST'])
def content_recommendations():
    data = request.get_json()
    query = data.get('title', '')

    if(query == ''):
        return jsonify({
            "error": True,
            "message": "Title cannot be empty."
        }), 400
    
    title = app_instance.movie_finder(query)
    top_n = data.get('top_n', 10)
    results = app_instance.run_content_recommender(title, top_n).to_dict(orient='records')
    
    # Aggiungi informazioni sul film originale
    response = {
        "original_movie": {
            "title": title
        },
        "recommendations": results
    }
    
    return jsonify(response)

@app.route('/recommend/content_mab', methods=['POST'])
def content_recommendations_mab():
    data = request.get_json()
    query = data.get('title', '')

    if(query == ''):
        return jsonify({
            "error": True,
            "message": "Title cannot be empty."
        }), 400
    
    title = app_instance.movie_finder(query)
    top_n = data.get('top_n', 10)
    results_ids = app_instance.run_content_recommender_mab(title)
    results = app_instance.df_with_abstracts[app_instance.df_with_abstracts['movieId'].isin(results_ids)].to_dict(orient='records')
    
    # Aggiungi informazioni sul film originale
    response = {
        "original_movie": {
            "title": title
        },
        "recommendations": results
    }
    
    return jsonify(response)

@app.route('/recommend/item', methods=['POST'])
def item_recommendations():
    data = request.get_json()
    movie_id = data.get('movie_id', 0)
    top_n = data.get('top_n', 10)

    if(movie_id <= 0):
        return jsonify({
            "error": True,
            "message": "Movie ID must be a positive integer."
        }), 404
    
    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts['movieId'] == movie_id]

    if len(movie) == 0:
        return jsonify({
            "error": True,
            "message": f"Movie with ID {movie_id} not found."
        }), 404
    
    # Ottieni i dettagli del film originale
    original_movie = app_instance.show_movie_details(movie_id)
    
    # Ottieni le raccomandazioni
    df = app_instance.run_collaborative_item_recommender(movie_id, top_n)
    df = df.reset_index()
    df_results = df.merge(app_instance.df_with_abstracts, on='movieId', suffixes=('', '_drop'))
    df_results = df_results.loc[:, ~df_results.columns.str.contains('_drop')]
    results = df_results.to_dict(orient='records')
    
    # Crea la risposta con il film originale e le raccomandazioni
    response = {
        "original_movie": {
            "id": movie_id,
            "title": original_movie.get('title', 'Unknown')
        },
        "recommendations": results
    }
    
    return jsonify(response)

@app.route('/recommend/item_mab', methods=['POST'])
def item_recommendations_mab():
    data = request.get_json()
    movie_id = data.get('movie_id', 0)
    top_n = data.get('top_n', 10)

    if(movie_id <= 0):
        return jsonify({
            "error": True,
            "message": "Movie ID must be a positive integer."
        }), 404
    
    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts['movieId'] == movie_id]
    if len(movie) == 0:
        return jsonify({
            "error": True,
            "message": f"Movie with ID {movie_id} not found."
        }), 404
    
    # Ottieni i dettagli del film originale
    original_movie = app_instance.show_movie_details(movie_id)
    
    # Ottieni le raccomandazioni
    results_ids = app_instance.run_collaborative_item_recommender_mab(movie_id, top_n)
    results = app_instance.df_with_abstracts[app_instance.df_with_abstracts['movieId'].isin(results_ids)].to_dict(orient='records')
    
    # Crea la risposta con il film originale e le raccomandazioni
    response = {
        "original_movie": {
            "id": movie_id,
            "title": original_movie.get('title', 'Unknown')
        },
        "recommendations": results
    }
    
    return jsonify(response)

@app.route('/recommend/user', methods=['POST'])
def user_recommendations():
    data = request.get_json()
    user_id = data.get('user_id', 0)

    if(user_id <= 0):
        return jsonify({
            "error": True,
            "message": "User ID must be a positive integer."
        }), 404
    
    if user_id > 610:
        return jsonify({
            "error": True,
            "message": f"User with ID {user_id} not found."
        }), 404
    
    top_n = data.get('top_n', 10)
    df = app_instance.run_collaborative_user_recommender(user_id, top_n)
    df = df.reset_index()
    df_results = df.merge(app_instance.df_with_abstracts, on='movieId', suffixes=('', '_drop'))
    df_results = df_results.loc[:, ~df_results.columns.str.contains('_drop')]
    df_results = df_results.drop(columns=['values'])
    json_response ={
        "userId": user_id,
        "results": df_results.to_dict(orient='records')
    }

    return jsonify(json_response)

@app.route('/recommend/user_mab', methods=['POST'])
def user_recommendations_mab():
    data = request.get_json()
    user_id = data.get('user_id', 0)

    if(user_id <= 0):
        return jsonify({
            "error": True,
            "message": "User ID must be a positive integer."
        }), 404
    
    if user_id > 610:
        return jsonify({
            "error": True,
            "message": f"User with ID {user_id} not found."
        }), 404

    top_n = data.get('top_n', 10)

    results_ids = app_instance.run_collaborative_user_recommender_mab(user_id, top_n)
    results = app_instance.df_with_abstracts[app_instance.df_with_abstracts['movieId'].isin(results_ids)].to_dict(orient='records')
    json_response ={
        "userId": user_id,
        "results": results
    }

    return jsonify(json_response)

@app.route('/recommend/director/movie', methods=['POST']) 
def director_recommendations_movie():
    data = request.get_json()
    movie_id = data.get('movie_id', 0)

    if(movie_id <= 0):
        return jsonify({
            "error": True,
            "message": "Movie ID must be a positive integer."
        }), 404
    
    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts['movieId'] == movie_id]
    if len(movie) == 0:
        return jsonify({
            "error": True,
            "message": f"Movie with ID {movie_id} not found."
        }), 404
    
    max_actors = data.get('max_actors', 5)
    results = app_instance.run_director_recommender_by_movie(movie_id, max_actors)
    return jsonify({'result': results})

@app.route('/recommend/director/name', methods=['POST'])
def director_recommendations_name():
    data = request.get_json()
    director = data.get('director', '')

    if director == '':
        return jsonify({
            "error": True,
            "message": "Director name cannot be empty."
        }), 400
    
    max_actors = data.get('max_actors', 5)
    results = app_instance.run_director_recommender_by_director(director, max_actors)
    return jsonify({'result': results})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = data.get('user_id', 0)
    movie_id = data.get('movie_id', 0)
    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts['movieId'] == movie_id]
    movie.reset_index(drop=True, inplace=True)

    if len(movie) == 0:
        return jsonify({
            "error": True,
            "message": f"Il film con ID {movie_id} non è stato trovato."
        }), 404
    
    # Verifica se l'utente esiste
    if user_id > 610:
        return jsonify({
            "error": True,
            "message": f"L'utente con ID {user_id} non esiste."
        }), 404

    ratings = app_instance.df_ratings[app_instance.df_ratings['movieId'] == movie_id]
    if len(ratings) > 0:
        avg_rating = ratings['rating'].mean()
        num_ratings = len(ratings)
        movie['avg_rating'] = avg_rating
        movie['num_ratings'] = num_ratings

    if(ratings[ratings['userId'] == user_id].shape[0] > 0):
        movie['already_rated'] = True
        movie['prediction_rating'] = ratings[ratings['userId'] == user_id]['rating'].values[0]
    else:
        movie['already_rated'] = False
        movie['prediction_rating'] = np.random.uniform(0, 5)

    movie = movie.to_dict(orient='records')
    json_response = {
        "userId": user_id,
        "movie": movie,
    }
    return jsonify(json_response)
    
if __name__ == '__main__':
    app.run(debug=True)