from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from cb_recommender import ContentBasedRecommender
from cf_recommender import CollaborativeRecommender
from sklearn.neighbors import NearestNeighbors
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
        except Exception as e:
            print(f"Errore nel caricamento dei dati base: {e}")

    def movie_finder(self, title):
        all_titles = self.df_movies['title'].tolist()
        closest_match = process.extractOne(title, all_titles)
        return closest_match[0]

    def initialize_content_recommender(self) -> None:
        if self.content_initialized:
            return
        try:
            df_with_abstracts = pd.read_csv(self.movies_with_abstracts_path)
            self.content_recommender = ContentBasedRecommender(df_with_abstracts)
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
        if self.df_movies is None:
            return []
        results = self.df_movies[self.df_movies['title'] == query]
        movies_list = []
        for _, row in results.iterrows():
            movies_list.append({'id': row.name, 'title': row['title'], 'genres': row['genres']})
        return movies_list

    def show_movie_details(self, movie_id: int) -> dict:
        if self.df_movies is None:
            return {}
        try:
            movie = self.df_movies.loc[movie_id].to_dict()
            if self.df_ratings is not None:
                ratings = self.df_ratings[self.df_ratings['movieId'] == movie_id]
                if len(ratings) > 0:
                    avg_rating = ratings['rating'].mean()
                    num_ratings = len(ratings)
                    movie['avg_rating'] = avg_rating
                    movie['num_ratings'] = num_ratings
                    movie['id'] = movie_id
            return movie
        except (IndexError, KeyError):
            return {}

    def run_content_recommender(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        if not self.content_initialized:
            self.initialize_content_recommender()
            if not self.content_initialized:
                return pd.DataFrame()
        return self.content_recommender.recommend(movie_title, top_n=top_n)

    def run_collaborative_item_recommender(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return pd.DataFrame()
        return self.collaborative_recommender.get_item_recommendations(movie_id, self.utility_matrix, self.df_movies).head(top_n)

    def run_collaborative_user_recommender(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return pd.DataFrame()
        return self.collaborative_recommender.get_user_recommendations(user_id, self.utility_matrix, self.df_movies).head(top_n)

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
    title = app_instance.movie_finder(query)
    results = app_instance.search_movie_by_title(title)
    return jsonify(results)

@app.route('/movie/<int:movie_id>', methods=['GET'])
def movie_details(movie_id):
    details = app_instance.show_movie_details(movie_id)
    print(details)
    return jsonify(details)

@app.route('/recommend/content', methods=['POST'])
def content_recommendations():
    data = request.get_json()
    query = data.get('query', '')
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

@app.route('/recommend/item', methods=['POST'])
def item_recommendations():
    data = request.get_json()
    movie_id = data.get('movie_id', 0)
    top_n = data.get('top_n', 10)
    
    # Ottieni i dettagli del film originale
    original_movie = app_instance.show_movie_details(movie_id)
    
    # Ottieni le raccomandazioni
    df = app_instance.run_collaborative_item_recommender(movie_id, top_n)
    df = df.reset_index()
    df.rename(columns={'movieId': 'id'}, inplace=True)
    results = df.to_dict(orient='records')
    
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
    top_n = data.get('top_n', 10)
    df = app_instance.run_collaborative_user_recommender(user_id, top_n)
    df = df.reset_index()
    df.rename(columns={'movieId': 'id'}, inplace=True)
    results = df.to_dict(orient='records')
    return jsonify(results)

@app.route('/recommend/director/movie', methods=['POST']) 
def director_recommendations_movie():
    data = request.get_json()
    movie_id = data.get('movie_id', 0)
    max_actors = data.get('max_actors', 5)
    results = app_instance.run_director_recommender_by_movie(movie_id, max_actors)
    return jsonify({'result': results})

@app.route('/recommend/director/name', methods=['POST'])
def director_recommendations_name():
    data = request.get_json()
    director = data.get('director', '')
    max_actors = data.get('max_actors', 5)
    results = app_instance.run_director_recommender_by_director(director, max_actors)
    return jsonify({'result': results})

if __name__ == '__main__':
    app.run(debug=True)