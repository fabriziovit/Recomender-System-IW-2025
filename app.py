# Import necessary libraries
import os
import logging
import numpy as np
import pandas as pd
from test_mab import mab
from flask_cors import CORS
from fuzzywuzzy import process
from epsilon_mab import EpsGreedyMAB
from latent_factor_model_recomm import MF_SGD_User_Based
from mab_on_collaborative_filtering import mab_on_collabfilter
from mab_on_content_based import mab_on_contentbased
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from content_based_recomm import ContentBasedRecommender
from collaborative_filtering_recomm import CollaborativeRecommender
from utils import load_movielens_data, log_epsilon_decay, exp_epsilon_decay, linear_epsilon_decay
from director_recomm import recommend_by_movie_id, recommend_films_with_actors

# Configure logger format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        self.sgd_model: MF_SGD_User_Based = None
        self.content_initialized = False
        self.collaborative_initialized = False
        self.director_initialized = False
        self.sgd_initialized = False
        self.movies_with_abstracts_path = "./dataset/movies_with_abstracts_complete.csv"  # Change path to a dynamic one
        self._load_basic_data()

    # Load MovieLens base datasets and utility matrix
    def _load_basic_data(self) -> None:
        try:
            self.df_movies, self.df_ratings, self.df_tags = load_movielens_data(self.data_dir)
            self.utility_matrix = self.df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
            self.df_with_abstracts = pd.read_csv(self.movies_with_abstracts_path, dtype={"imdbId": str, "tmdbId": str})
        except Exception as e:
            logging.info(f"Error loading base data: {e}")

    # Use fuzzy matching to find the closest movie title
    def movie_finder(self, title):
        all_titles = self.df_with_abstracts["title"].tolist()
        closest_match = process.extractOne(title, all_titles)
        return closest_match[0]

    # Initialize the content-based recommender
    def initialize_content_recommender(self) -> None:
        if self.content_initialized:
            return
        try:
            self.content_recommender = ContentBasedRecommender(self.df_with_abstracts)
            self.content_initialized = True
        except Exception as e:
            logging.info(f"Error initializing content recommender: {e}")

    # Initialize the collaborative recommender with KNN models
    def initialize_collaborative_recommender(self, n_neighbors: int = 20) -> None:
        if self.collaborative_initialized:
            return
        try:
            knn_model_item = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors + 1, n_jobs=-1)
            knn_model_user = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors + 1, n_jobs=-1)
            self.collaborative_recommender = CollaborativeRecommender(knn_model_item, knn_model_user)
            self.collaborative_recommender.fit_item_model(self.utility_matrix)
            self.collaborative_recommender.fit_user_model(self.utility_matrix)
            self.collaborative_initialized = True
        except Exception as e:
            logging.info(f"Error initializing collaborative recommender: {e}")

    # Load pretrained matrix factorization model
    def initialize_sgd_recommender(self) -> None:
        self.sgd_model = MF_SGD_User_Based.load_model("models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl")
        self.sgd_initialized = True

    # Check if the director-based recommender can be used
    def check_director_recommender(self) -> bool:
        if os.path.exists(self.movies_with_abstracts_path):
            self.df_with_abstracts = pd.read_csv(self.movies_with_abstracts_path)
            self.director_initialized = True
            return True
        else:
            return False

    # Search for movies with titles containing a specific query
    def search_movie_by_title(self, query: str) -> list:
        if self.df_with_abstracts is None:
            return []
        results = self.df_with_abstracts[self.df_with_abstracts["title"].str.contains(query, case=False, regex=False)].fillna("Data not found")
        movies_list = []
        for _, row in results.iterrows():
            movies_list.append(row)
        return movies_list

    # Return movie details including average rating and number of ratings
    def show_movie_details(self, movie_id: int) -> dict:
        if self.df_with_abstracts is None:
            return {}
        try:
            movie = self.df_with_abstracts[self.df_with_abstracts["movieId"] == movie_id].iloc[0].fillna("Data Not Found").to_dict()
            if self.df_ratings is not None:
                ratings = self.df_ratings[self.df_ratings["movieId"] == movie_id]
                if len(ratings) > 0:
                    avg_rating = ratings["rating"].mean()
                    num_ratings = len(ratings)
                    movie["avg_rating"] = avg_rating
                    movie["num_ratings"] = num_ratings
            return movie
        except (IndexError, KeyError):
            return {}

    # Run content-based recommendations
    def run_content_recommender(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        if not self.content_initialized:
            self.initialize_content_recommender()
            if not self.content_initialized:
                return pd.DataFrame()
        return self.content_recommender.recommend(movie_title, top_n=top_n)

    # Run content-based recommendations with Multi-Armed Bandit
    def run_content_recommender_mab(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        if not self.content_initialized:
            self.initialize_content_recommender()
            if not self.content_initialized:
                return pd.DataFrame()
        return mab_on_contentbased(movie_title, self.df_ratings, recommender=self.content_recommender)[:top_n]

    # Run item-based collaborative recommendations
    def run_collaborative_item_recommender(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return pd.DataFrame()
        return self.collaborative_recommender.get_item_recommendations(movie_id, self.df_movies).head(top_n)

    # Run item-based collaborative recommendations with MAB
    def run_collaborative_item_recommender_mab(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return pd.DataFrame()
        return mab_on_collabfilter(self.df_ratings, self.df_movies, movie_id, None, recommender=self.collaborative_recommender, utility_matrix=self.utility_matrix)[:top_n]

    # Run user-based collaborative recommendations
    def run_collaborative_user_recommender(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return pd.DataFrame()
        return self.collaborative_recommender.get_user_recommendations(user_id, self.df_movies).head(top_n)

    # Run user-based collaborative recommendations with MAB
    def run_collaborative_user_recommender_mab(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return pd.DataFrame()
        return mab_on_collabfilter(self.df_ratings, self.df_movies, None, user_id, recommender=self.collaborative_recommender, utility_matrix=self.utility_matrix)[:top_n]

    # Run director-based recommendations using a selected movie
    def run_director_recommender_by_movie(self, movie_id: int, max_actors: int = 5) -> str:
        if not self.check_director_recommender():
            return "Director recommender not available."
        return recommend_by_movie_id(self.movies_with_abstracts_path, movie_id, max_actors=max_actors, movie_title_selected=True)

    # Run director-based recommendations using a director's name
    def run_director_recommender_by_director(self, director: str, max_actors: int = 5) -> str:
        if not self.check_director_recommender():
            return "Director recommender not available."
        return recommend_films_with_actors(director, max_actors=max_actors, movie_title_selected=False)

    # Run matrix factorization (SGD-based) recommendations
    def run_sgd_recommendations(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.sgd_initialized:
            self.initialize_sgd_recommender()
            if not self.sgd_initialized:
                return pd.DataFrame()
        return self.sgd_model.get_recommendations(self.utility_matrix, user_id=user_id).head(top_n)

    # Run MAB with log epsilon decay on SGD model
    def run_mab_sgd_model_log_epsilon_decay(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if not self.sgd_initialized:
            self.initialize_sgd_recommender()
            if not self.sgd_initialized:
                return pd.DataFrame()
        df_expected = self.sgd_model.get_recommendations(self.utility_matrix, user_id)
        df_expected = df_expected.merge(self.df_movies, on="movieId")[["title", "values"]].sort_index(ascending=True)
        # Set the index to make it compatible with the bandit
        new_index = pd.RangeIndex(df_expected.shape[0], name="idxarm")
        df_expected.reset_index(drop=False, inplace=True)
        df_expected.set_index(new_index, inplace=True)

        bandit_mab = EpsGreedyMAB(n_arms=df_expected.shape[0], epsilon=0.9, Q0=0.0)  #! check epsilon value
        bandit_mab.set_epsilon_deacy(log_epsilon_decay)

        top_k = mab(df_expected, bandit_mab, num_rounds=10_000)[:top_n]
        return top_k

    # Run MAB with exponential epsilon decay on SGD model
    def run_mab_sgd_model_exp_epsilon_decay(self, user_id: int, top_n: int = 10) -> pd.DataFrame:  # To add API call
        if not self.sgd_initialized:
            self.initialize_sgd_recommender()
            if not self.sgd_initialized:
                return pd.DataFrame()
        df_expected = self.sgd_model.get_recommendations(self.utility_matrix, user_id)
        df_expected = df_expected.merge(self.df_movies, on="movieId")[["title", "values"]].sort_index(ascending=True)
        # Set the index to make it compatible with the bandit
        new_index = pd.RangeIndex(df_expected.shape[0], name="idxarm")
        df_expected.reset_index(drop=False, inplace=True)
        df_expected.set_index(new_index, inplace=True)

        bandit_mab = EpsGreedyMAB(n_arms=df_expected.shape[0], epsilon=0.9, Q0=0.0)  #! check epsilon value
        bandit_mab.set_epsilon_deacy(exp_epsilon_decay)

        top_k = mab(df_expected, bandit_mab, num_rounds=10_000)[:top_n]
        return top_k

    # Run MAB with linear epsilon decay on SGD model
    def run_mab_sgd_model_linear_epsilon_decay(self, user_id: int, top_n: int = 10) -> pd.DataFrame:  # To add API call
        if not self.sgd_initialized:
            self.initialize_sgd_recommender()
            if not self.sgd_initialized:
                return pd.DataFrame()
        df_expected = self.sgd_model.get_recommendations(self.utility_matrix, user_id)
        df_expected = df_expected.merge(self.df_movies, on="movieId")[["title", "values"]].sort_index(ascending=True)
        # Set the index to make it compatible with the bandit
        new_index = pd.RangeIndex(df_expected.shape[0], name="idxarm")
        df_expected.reset_index(drop=False, inplace=True)
        df_expected.set_index(new_index, inplace=True)

        bandit_mab = EpsGreedyMAB(n_arms=df_expected.shape[0], epsilon=0.9, Q0=0.0)  #! check epsilon value
        bandit_mab.set_epsilon_deacy(linear_epsilon_decay)

        top_k = mab(df_expected, bandit_mab, num_rounds=10_000)[:top_n]
        return top_k

    # Run MAB with fixed epsilon (no decay)
    def run_mab_fixed_epsilon(self, user_id: int, top_n: int = 10, epsilon: float = 0.1) -> pd.DataFrame:
        if not self.sgd_initialized:
            self.initialize_sgd_recommender()
            if not self.sgd_initialized:
                return pd.DataFrame()
        df_expected = self.sgd_model.get_recommendations(self.utility_matrix, user_id)
        df_expected = df_expected.merge(self.df_movies, on="movieId")[["title", "values"]].sort_index(ascending=True)
        # Set the index to make it compatible with the bandit
        new_index = pd.RangeIndex(df_expected.shape[0], name="idxarm")
        df_expected.reset_index(drop=False, inplace=True)
        df_expected.set_index(new_index, inplace=True)

        bandit_mab = EpsGreedyMAB(n_arms=df_expected.shape[0], epsilon=epsilon, Q0=0.0)

        top_k = mab(df_expected, bandit_mab, num_rounds=10_000, decay=False)[:top_n]
        return top_k

    # Predict rating for a given user/movie using SGD model
    def run_sgd_predictions(self, user_id: int, utility_matrix: pd.DataFrame) -> float:
        if not self.sgd_initialized:
            self.initialize_sgd_recommender()
            if not self.sgd_initialized:
                return 0.0
        return self.sgd_model.get_recommendations(utility_matrix, user_id)

    # Predict rating for a given user/movie using KNN (collaborative filtering)
    def run_knn_predictions(self, user_id: int, movie_id: int) -> float:
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                return 0.0
        return self.collaborative_recommender.get_prediction_value_clipped(user_id, movie_id)


app_instance = MovieRecommenderApp()


@app.route("/search", methods=["POST"])
def search_movies():
    data = request.get_json()
    query = data.get("query", "")
    results = app_instance.search_movie_by_title(query)
    results = [result.to_dict() for result in results]
    return jsonify(results)


@app.route("/movie/<int:movie_id>", methods=["GET"])
def movie_details(movie_id):
    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]

    if movie_id <= 0:
        return jsonify({"error": True, "message": "Movie ID must be a positive integer."}), 404

    if len(movie) == 0:
        return jsonify({"error": True, "message": f"Movie with ID {movie_id} not found."}), 404

    details = app_instance.show_movie_details(movie_id)
    return jsonify(details)


@app.route("/user/<int:user_id>", methods=["GET"])
def user_movies_list(user_id):
    if user_id <= 0:
        return jsonify({"error": True, "message": "User ID must be a positive integer."}), 404

    if user_id > 610:
        return jsonify({"error": True, "message": f"User with ID {user_id} not found."}), 404

    user_ratings = app_instance.df_ratings[app_instance.df_ratings["userId"] == user_id]
    user_movies = user_ratings.merge(app_instance.df_with_abstracts, on="movieId", suffixes=("", "_drop")).fillna("Data not found")
    user_movies = user_movies.drop(columns=["dbpedia_abstract", "dbpedia_director"])
    user_movies = user_movies.loc[:, ~user_movies.columns.str.contains("_drop")]
    user_movies = user_movies.to_dict(orient="records")

    json_response = {"userId": user_id, "results": user_movies}
    return jsonify(json_response)


#####Recommendation Endpoints#####


@app.route("/recommend/content", methods=["POST"])
def content_recommendations():
    data = request.get_json()
    query = data.get("title", "")

    if query == "":
        return jsonify({"error": True, "message": "Title cannot be empty."}), 400

    title = app_instance.movie_finder(query)
    top_n = data.get("top_n", 10)
    results = app_instance.run_content_recommender(title, top_n).to_dict(orient="records")

    # Add original movie information
    response = {"original_movie": {"title": title}, "recommendations": results}

    return jsonify(response)


@app.route("/recommend/content_mab", methods=["POST"])
def content_recommendations_mab():
    data = request.get_json()
    query = data.get("title", "")

    if query == "":
        return jsonify({"error": True, "message": "Title cannot be empty."}), 400

    title = app_instance.movie_finder(query)
    top_n = data.get("top_n", 10)
    results_ids = app_instance.run_content_recommender_mab(title)
    ordered_results = []

    for movie_id in results_ids:
        movie_row = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]
        if not movie_row.empty:
            # Convert the DataFrame row to a dict
            movie_dict = movie_row.iloc[0].to_dict()
            # Convert all NumPy values to standard Python types
            ordered_results.append({k: v.item() if isinstance(v, np.number) else v for k, v in movie_dict.items()})

    # The ordered_results list now contains the dictionaries in the original order
    results = ordered_results

    # Add original movie information
    response = {"original_movie": {"title": title}, "recommendations": results}

    return jsonify(response)


@app.route("/recommend/item", methods=["POST"])
def item_recommendations():
    data = request.get_json()
    movie_id = data.get("movie_id", 0)
    top_n = data.get("top_n", 10)

    if movie_id <= 0:
        return jsonify({"error": True, "message": "Movie ID must be a positive integer."}), 404

    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]

    if len(movie) == 0:
        return jsonify({"error": True, "message": f"Movie with ID {movie_id} not found."}), 404

    # Get the original movie details
    original_movie = app_instance.show_movie_details(movie_id)

    # Get the recommendations
    df = app_instance.run_collaborative_item_recommender(movie_id, top_n)
    df = df.reset_index()
    df_results = df.merge(app_instance.df_with_abstracts, on="movieId", suffixes=("", "_drop"))
    df_results = df_results.loc[:, ~df_results.columns.str.contains("_drop")]
    results = df_results.to_dict(orient="records")

    # Create the response with the original movie and recommendations
    response = {"original_movie": {"id": movie_id, "title": original_movie.get("title", "Unknown")}, "recommendations": results}

    return jsonify(response)


@app.route("/recommend/item_mab", methods=["POST"])
def item_recommendations_mab():
    data = request.get_json()
    movie_id = data.get("movie_id", 0)
    top_n = data.get("top_n", 10)

    if movie_id <= 0:
        return jsonify({"error": True, "message": "Movie ID must be a positive integer."}), 404

    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]
    if len(movie) == 0:
        return jsonify({"error": True, "message": f"Movie with ID {movie_id} not found."}), 404

    # Get the original movie details
    original_movie = app_instance.show_movie_details(movie_id)

    # Get the recommendations
    results_ids = app_instance.run_collaborative_item_recommender_mab(movie_id, top_n)

    # Convert results_ids to a list of standard Python integers if it's a NumPy array
    if isinstance(results_ids, np.ndarray):
        results_ids = results_ids.tolist()

    ordered_results = []

    # Iterate through the IDs in the original order
    for movie_id in results_ids:
        movie_row = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]
        if not movie_row.empty:
            # Convert the DataFrame row to a dict
            movie_dict = movie_row.iloc[0].to_dict()
            # Convert all NumPy values to standard Python types
            ordered_results.append({k: v.item() if isinstance(v, np.number) else v for k, v in movie_dict.items()})

    # Create the response with the original movie and recommendations
    response = {
        "original_movie": {"id": movie_id if not isinstance(movie_id, np.number) else movie_id.item(), "title": original_movie.get("title", "Unknown")},
        "recommendations": ordered_results,
    }

    return jsonify(response)


@app.route("/recommend/user", methods=["POST"])
def user_recommendations():
    data = request.get_json()
    user_id = data.get("user_id", 0)

    if user_id <= 0:
        return jsonify({"error": True, "message": "User ID must be a positive integer."}), 404

    if user_id > 610:
        return jsonify({"error": True, "message": f"User with ID {user_id} not found."}), 404

    top_n = data.get("top_n", 10)
    df = app_instance.run_collaborative_user_recommender(user_id, top_n)
    df = df.reset_index()
    df_results = df.merge(app_instance.df_with_abstracts, on="movieId", suffixes=("", "_drop"))
    df_results = df_results.loc[:, ~df_results.columns.str.contains("_drop")]
    df_results = df_results.drop(columns=["values"])
    json_response = {"userId": user_id, "results": df_results.to_dict(orient="records")}

    logging.info(f"json_response: {json_response}")

    return jsonify(json_response)


@app.route("/recommend/user_mab", methods=["POST"])
def user_recommendations_mab():
    data = request.get_json()
    user_id = data.get("user_id", 0)

    if user_id <= 0:
        return jsonify({"error": True, "message": "User ID must be a positive integer."}), 404

    if user_id > 610:
        return jsonify({"error": True, "message": f"User with ID {user_id} not found."}), 404

    top_n = data.get("top_n", 10)

    results_ids = app_instance.run_collaborative_user_recommender_mab(user_id, top_n)
    ordered_results = []

    for movie_id in results_ids:
        movie_row = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]
        if not movie_row.empty:
            # Convert the DataFrame row to a dict
            movie_dict = movie_row.iloc[0].to_dict()
            # Convert all NumPy values to standard Python types
            ordered_results.append({k: v.item() if isinstance(v, np.number) else v for k, v in movie_dict.items()})

    # The ordered_results list now contains the dictionaries in the original order
    results = ordered_results
    json_response = {"userId": user_id, "results": results}

    return jsonify(json_response)


@app.route("/recommend/director/movie", methods=["POST"])
def director_recommendations_movie():
    data = request.get_json()
    movie_id = data.get("movie_id", 0)

    if movie_id <= 0:
        return jsonify({"error": True, "message": "Movie ID must be a positive integer."}), 404

    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]
    if len(movie) == 0:
        return jsonify({"error": True, "message": f"Movie with ID {movie_id} not found."}), 404

    max_actors = data.get("max_actors", 5)
    results = app_instance.run_director_recommender_by_movie(movie_id, max_actors)
    return jsonify({"result": results})


@app.route("/recommend/director/name", methods=["POST"])
def director_recommendations_name():
    data = request.get_json()
    director = data.get("director", "")

    if director == "":
        return jsonify({"error": True, "message": "Director name cannot be empty."}), 400

    max_actors = data.get("max_actors", 5)
    results = app_instance.run_director_recommender_by_director(director, max_actors)
    return jsonify({"result": results})


@app.route("/recommend/sgd", methods=["POST"])
def sgd_recommendations():
    data = request.get_json()
    user_id = data.get("user_id", 0)
    top_n = data.get("top_n", 10)

    if user_id <= 0:
        return jsonify({"error": True, "message": "User ID must be a positive integer."}), 404

    if user_id > 610:
        return jsonify({"error": True, "message": f"User with ID {user_id} not found."}), 404

    results_ids = app_instance.run_sgd_recommendations(user_id, top_n).index.tolist()
    ordered_results = []

    # Iterate through the IDs in the original order
    for movie_id in results_ids:
        movie_row = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]
        if not movie_row.empty:
            # Convert the DataFrame row to a dict
            movie_dict = movie_row.iloc[0].to_dict()
            # Convert all NumPy values to standard Python types
            ordered_results.append({k: v.item() if isinstance(v, np.number) else v for k, v in movie_dict.items()})

    results = ordered_results
    json_response = {"userId": user_id, "results": results}
    return jsonify(json_response)


@app.route("/recommend/sgd_mab_exp", methods=["POST"])
def mab_sgd_exp_epsilon_recommendations():
    data = request.get_json()
    user_id = data.get("user_id", 0)
    top_n = data.get("top_n", 10)

    if user_id <= 0:
        return jsonify({"error": True, "message": "User ID must be a positive integer."}), 404

    if user_id > 610:
        return jsonify({"error": True, "message": f"User with ID {user_id} not found."}), 404

    results_ids = app_instance.run_mab_sgd_model_exp_epsilon_decay(user_id, top_n)

    ordered_results = []

    # Iterate through the IDs in the original order
    for movie_id in results_ids:
        movie_row = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]
        if not movie_row.empty:
            # Convert the DataFrame row to a dict
            movie_dict = movie_row.iloc[0].to_dict()
            # Convert all NumPy values to standard Python types
            ordered_results.append({k: v.item() if isinstance(v, np.number) else v for k, v in movie_dict.items()})

    # The ordered_results list now contains the dictionaries in the original order
    results = ordered_results
    json_response = {"userId": user_id, "results": results}
    return jsonify(json_response)


@app.route("/recommend/sgd_mab_fixed", methods=["POST"])
def mab_sgd_fixed_epsilon_recommendations():
    data = request.get_json()
    user_id = data.get("user_id", 0)
    top_n = data.get("top_n", 10)
    epsilon = data.get("epsilon", 0.1)
    epsilon = float(epsilon)  # Ensure epsilon is a float

    if user_id <= 0:
        return jsonify({"error": True, "message": "User ID must be a positive integer."}), 404

    if user_id > 610:
        return jsonify({"error": True, "message": f"User with ID {user_id} not found."}), 404

    results_ids = app_instance.run_mab_fixed_epsilon(user_id, top_n, epsilon)
    logging.info(f"results_ids: {results_ids}")

    ordered_results = []

    # Iterate through the IDs in the original order
    for movie_id in results_ids:
        movie_row = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id]
        if not movie_row.empty:
            # Convert the DataFrame row to a dict
            movie_dict = movie_row.iloc[0].to_dict()
            # Convert all NumPy values to standard Python types
            ordered_results.append({k: v.item() if isinstance(v, np.number) else v for k, v in movie_dict.items()})

    # The ordered_results list now contains the dictionaries in the original order
    results = ordered_results
    json_response = {"userId": user_id, "results": results}
    return jsonify(json_response)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_id = data.get("user_id", 0)
    movie_id = data.get("movie_id", 0)
    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id].copy()
    movie.reset_index(drop=True, inplace=True)

    if len(movie) == 0:
        return jsonify({"error": True, "message": f"Movie with ID {movie_id} not found."}), 404

    # Check if the user exists
    if user_id > 610:
        return jsonify({"error": True, "message": f"User with ID {user_id} does not exist."}), 404

    ratings = app_instance.df_ratings[app_instance.df_ratings["movieId"] == movie_id]
    if len(ratings) > 0:
        avg_rating = ratings["rating"].mean()
        num_ratings = len(ratings)
        movie["avg_rating"] = avg_rating
        movie["num_ratings"] = num_ratings

    if ratings[ratings["userId"] == user_id].shape[0] > 0:
        movie["already_rated"] = True
        movie["prediction_rating"] = ratings[ratings["userId"] == user_id]["rating"].values[0]
    else:
        movie["already_rated"] = False
        df_recoomendations = app_instance.run_sgd_predictions(user_id, app_instance.utility_matrix)

        movie["prediction_rating"] = np.clip(df_recoomendations.loc[movie_id].values[0], 0.5, 5.0)

    movie = movie.to_dict(orient="records")
    json_response = {
        "userId": user_id,
        "movie": movie,
    }
    return jsonify(json_response)


@app.route("/predict_knn", methods=["POST"])
def predict_knn():
    data = request.get_json()
    user_id = data.get("user_id", 0)
    movie_id = data.get("movie_id", 0)
    movie = app_instance.df_with_abstracts[app_instance.df_with_abstracts["movieId"] == movie_id].copy()
    movie.reset_index(drop=True, inplace=True)

    if len(movie) == 0:
        return jsonify({"error": True, "message": f"Movie with ID {movie_id} not found."}), 404

    # Check if the user exists
    if user_id > 610:
        return jsonify({"error": True, "message": f"User with ID {user_id} does not exist."}), 404

    ratings = app_instance.df_ratings[app_instance.df_ratings["movieId"] == movie_id]
    if len(ratings) > 0:
        avg_rating = ratings["rating"].mean()
        num_ratings = len(ratings)
        movie["avg_rating"] = avg_rating
        movie["num_ratings"] = num_ratings

    if ratings[ratings["userId"] == user_id].shape[0] > 0:
        movie["already_rated"] = True
        movie["prediction_rating"] = ratings[ratings["userId"] == user_id]["rating"].values[0]
    else:
        movie["already_rated"] = False
        movie["prediction_rating"] = app_instance.run_knn_predictions(user_id, movie_id)

    movie = movie.to_dict(orient="records")
    json_response = {
        "userId": user_id,
        "movie": movie,
    }
    return jsonify(json_response)


if __name__ == "__main__":
    app.run(debug=True)
