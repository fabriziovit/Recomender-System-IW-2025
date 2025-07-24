import time
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CollaborativeRecommender:

    def __init__(self, model_item: NearestNeighbors, model_user: NearestNeighbors):
        """Initialize the Recommender with a machine learning model."""
        if not isinstance(model_item, NearestNeighbors):
            raise ValueError(f"model_item must be NearestNeighbors")
        self.model_item = model_item

        if not isinstance(model_user, NearestNeighbors):
            raise ValueError(f"model_user must be NearestNeighbors")
        if model_user.metric != "cosine":
            raise ValueError("model_user metric must be 'cosine' for mean-centered approach.")
        self.model_user = model_user

        self.utility_matrix = None
        self.centered_utility_matrix = None
        self.user_means = None  # Store user means

        self._transposed_matrix = None
        self._is_fitted_on_matrix_T_item = False
        self._is_fitted_on_matrix_user = False
        self.dist_users = None
        self.dist_items = None

    def _compute_mean_centered_matrix_user_based(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate the mean-centered matrix, ignoring 0.0 for mean calculation."""
        # Calculate the mean for each user not considering 0.0 (replacing 0.0 with NaN)
        self.user_means = matrix.replace(0.0, np.nan).mean(axis=1)
        # Replace NaN values with 0.0 (absence of rating)
        self.user_means.fillna(0.0, inplace=True)

        # Subtract the user mean from each valid rating.
        # .sub() subtracts the Series user_means row by row.
        # .where() applies the subtraction only where valid_mask is True, otherwise puts 0.
        valid_mask = matrix > 0.0
        cenetered_matrix = matrix.sub(self.user_means, axis=0).where(valid_mask, 0)
        logging.info("# Mean-centered matrix calculated.")
        return cenetered_matrix

    def fit_item_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """Train the item-based model on matrix.T if necessary or if forced."""
        if not self._is_fitted_on_matrix_T_item or re_fit:
            self._transposed_matrix = matrix.T
            self.model_item.fit(self._transposed_matrix)
            self._is_fitted_on_matrix_T_item = True
            logging.info("# Model model_item trained on matrix.T (movies x users).")
        else:
            logging.info("# Model model_item already trained on matrix.T (movies x users): Skipping training.")

    def fit_user_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """Train the user-based model on matrix if necessary or if forced."""

        # NOTE: The model is fitted on the mean-centered matrix using the cosine metric
        # to calculate distances between users which is equivalent to calculating the Pearson Correlation
        # when calculating predictions using the _predict_rating method.

        if not self._is_fitted_on_matrix_user or re_fit:
            # 1. Store the utility_matrix
            self.utility_matrix = matrix
            # 2. Calculate and Store the mean-centered matrix
            self.centered_utility_matrix = self._compute_mean_centered_matrix_user_based(self.utility_matrix)
            logging.info("# Utility_matrix and mean-centered matrix calculated.")
            # 3. Train the user-based model on the centered matrix
            self.model_user.fit(self.centered_utility_matrix)
            self._is_fitted_on_matrix_user = True
            logging.info("# Model model_user trained on matrix_centered (users x movies).")
        else:
            logging.info("# Model model_user already trained on matrix_centered (users x movies): Skipping training.")

    # ************************************************************************************************ #

    def get_item_recommendations(self, movie_id: int, df_movies: pd.DataFrame) -> pd.DataFrame:
        """Recommend movies similar to a given movie using item-based collaborative filtering."""

        if not self._is_fitted_on_matrix_T_item or self._transposed_matrix is None:
            raise RuntimeError("Item model not trained or matrix not available.")
        if movie_id not in self._transposed_matrix.index:
            logging.info(f"Movie ID {movie_id} not found in matrix.")
            raise ValueError(f"Movie ID {movie_id} not found in matrix.")

        all_movie_ids = self._transposed_matrix.index

        # Select the features of the specified movie from the transposed matrix (movieId x userId)
        movie_features = self._transposed_matrix.loc[movie_id].values.reshape(1, -1)

        # Find the most similar movies using the NearestNeighbors model
        distances, pos_indexes = self.model_item.kneighbors(movie_features)

        logging.info(f"Distances for movie {movie_id}: {distances.squeeze().tolist()[1:]}")
        logging.info(f"Positional indices for movie {movie_id}: {pos_indexes.squeeze().tolist()[1:]}")

        # Derive the IDs of similar movies using positional indices
        similar_movies_list: list = [all_movie_ids[pos_idx] for pos_idx in pos_indexes.squeeze().tolist()[1:]]
        logging.info(f"Similar movies found for movie {movie_id}: {similar_movies_list}")

        # Save the item-item distances for the specified movie
        self.dist_items = pd.Series(distances.squeeze().tolist()[1:], index=similar_movies_list)
        logging.info(
            f"\nSimilar movies found for movie: {df_movies.loc[movie_id, 'title']} with distances: " + f"{[str(uid) + ': ' + str(val) for uid, val in self.dist_items.items()]}"
        )

        # Create a DataFrame with movie recommendations
        return df_movies.loc[similar_movies_list]

    # ************************************************************************************************ #

    def _get_recommendation_values(self, user_id: int) -> pd.Series:
        """Calculate recommendations for a specific user."""

        # 1. Pre-calculate similar users and their means once
        similar_users_ids: list = self._get_similar_users(user_id)  # Use centered model/matrix

        # Pre-filter movies to predict
        user_ratings = self.utility_matrix.loc[user_id]

        # Exclude movies already seen by the user
        seen_movies_ids = set(user_ratings[user_ratings > 0.0].index)
        movies_to_predict = [movie_id for movie_id in self.utility_matrix.columns if movie_id not in seen_movies_ids]

        # 3. Calculate predictions passing similar users and their pre-calculated means
        predicted_ratings_list = []
        for movie_id in movies_to_predict:
            pred = self._predict_rating(user_id, movie_id, similar_users_ids)
            predicted_ratings_list.append({"movieId": movie_id, "predicted_rating": pred})

        # 4. Create and sort the predictions DataFrame
        predictions_df = pd.DataFrame(predicted_ratings_list).set_index("movieId").rename(columns={"predicted_rating": "values"})
        return predictions_df.sort_values("values", ascending=False)

    def get_user_recommendations(self, user_id: int, df_movies: pd.DataFrame = None) -> pd.DataFrame:
        """Return recommendations for a specific user."""

        if not self._is_fitted_on_matrix_user or self.utility_matrix is None or self.centered_utility_matrix is None:
            raise RuntimeError("User model not trained or matrix not available.")
        if user_id not in self.utility_matrix.index:
            raise ValueError(f"User ID {user_id} not found in matrix.")

        # Calculate predictions for the specified user
        predictions_df = self._get_recommendation_values(user_id)

        if df_movies is None:
            # If no movie DataFrame is provided, return only predictions
            return predictions_df

        # Merge recommendations with movie DataFrame on movieId
        # logging.info(f"Predictions for user {user_id}:\n{predictions_df.merge(df_movies, how='left', on='movieId')}")
        return predictions_df.merge(df_movies, how="left", on="movieId")

    # ************************************************************************************************ #
    def _get_similar_users(self, user_id: int) -> list:
        """Get similar users using the KNN model trained on the mean-centered matrix."""

        # Ensure models are trained and matrices available
        if not self._is_fitted_on_matrix_user or self.utility_matrix is None or self.centered_utility_matrix is None:
            raise RuntimeError("User model not trained or matrices not available.")

        # 1. Get the CENTERED feature vector for the target user
        user_centered_feature = self.centered_utility_matrix.loc[user_id].values.reshape(1, -1)

        # 2. Find neighbors based on cosine distance in centered space
        distances, pos_indexes = self.model_user.kneighbors(user_centered_feature)

        # Indices refer to the order of the matrix (original or centered, they are the same)
        all_user_ids = self.utility_matrix.index
        similar_users_ids = all_user_ids[pos_indexes.squeeze()[1:]]

        # Save the COSINE user-user distances for the specified user
        # Cosine distance is between 0 (identical) and 2 (opposite), 1 (orthogonal)
        self.dist_users = pd.Series(distances.squeeze()[1:], index=similar_users_ids)

        return similar_users_ids

    def _predict_rating(self, user_id: int, movie_id: int, similar_users_ids=None) -> float:
        """
        Predict the rating for a user and movie using knn and mean centering.
        Uses the ORIGINAL MATRIX for ratings and means, but neighbors and their
        distances (cosine) were found using the centered matrix.
        """

        if user_id not in self.utility_matrix.index:
            raise ValueError(f"User ID {user_id} not found in the original matrix.")
        if movie_id not in self.utility_matrix.columns:
            raise ValueError(f"Movie ID {movie_id} not found in the original matrix.")

        # 1. Retrieve the mean of the target user
        target_user_mean = self.user_means.loc[user_id]
        if target_user_mean is None:
            raise ValueError(f"Mean rating for user ID {user_id} not found.")

        # 2. Calculate similar users if not available
        if similar_users_ids is None:
            similar_users_ids = self._get_similar_users(user_id)

        if len(similar_users_ids) <= 0:
            logging.info(f"No similar users found for user {user_id}.")
            return target_user_mean

        else:
            # 3. Retrieve the means of similar users calculated in fit_user_model
            similar_mean_values = self.user_means.loc[similar_users_ids]

            # 4. Retrieve the ratings of similar users
            similar_users_df = self.utility_matrix.loc[similar_users_ids]

            # 5. Retrieve the ratings of similar users for the specified movie (movie_id)
            similar_users_for_movie = similar_users_df[movie_id]
            valid_mask_for_movie = similar_users_for_movie > 0.0

            # If there is at least one valid rating > 0.0 among neighbors for this movie
            if valid_mask_for_movie.any():
                # Select valid ratings of neighbors for the movie
                valid_ratings = similar_users_for_movie[valid_mask_for_movie]

                # Select corresponding cosine distances for valid neighbors
                valid_distances = self.dist_users[valid_mask_for_movie]

                # Calculate cosine similarities (1 - cosine distance)
                valid_similarities = 1.0 - valid_distances

                # Select means of valid neighbors
                valid_similar_means = similar_mean_values[valid_mask_for_movie]

                # Calculate weighted sum: similarity * (neighbor_rating - neighbor_mean)
                weighted_sum = np.sum(valid_similarities * (valid_ratings - valid_similar_means))

                # Sum of absolute values of similarities as denominator
                sum_of_similarities = np.sum(np.abs(valid_similarities))

                # Avoid division by zero
                if sum_of_similarities == 0:
                    prediction_centered = 0  # No deviation if similarities are all zero
                else:
                    prediction_centered = weighted_sum / sum_of_similarities

                # Final prediction = target_user_mean + prediction_centered
                return target_user_mean + prediction_centered

            else:
                # If there are no valid ratings for this movie, return the target user's mean
                # logging.info(f"No valid ratings found for movie {movie_id} from similar users.")
                return target_user_mean

    def get_prediction_value_clipped(self, user_id: int, movie_id: int, min_rating: float = 0.5, max_rating: float = 5.0) -> float:
        """Calculate the prediction for a specific user and movie. (clipped between 0.5 and 5.0)"""
        predicted_rating = self._predict_rating(user_id, movie_id, None)
        return np.clip(predicted_rating, min_rating, max_rating)

    # ************************************************************************************************ #
    def evaluate_mae_rmse(self, test_matrix: pd.DataFrame, min_rating: float = 0.5, max_rating: float = 5.0):
        """Evaluate the model on the test set calculating MAE and RMSE."""

        # Ensure the user model has been trained
        if not self._is_fitted_on_matrix_user or self.utility_matrix is None:
            raise RuntimeError("User model not trained. Call fit_user_model on the training set first.")

        true_ratings = []
        predicted_ratings = []
        skipped_count = 0
        start_time = time.time()

        # *** --- Efficient Iteration on Test Set --- ***#
        # We use stack() to get a Series with (userId, movieId) as index and rating as value,
        # automatically ignoring zeros.
        test_ratings_series = test_matrix[test_matrix > 0].stack()
        total_ratings_to_predict = len(test_ratings_series)

        logging.info(f"# Starting evaluation on {total_ratings_to_predict} ratings in test set...")

        # Iterate on the resulting Series
        for idx, true_rating in test_ratings_series.items():
            logging.info(f"Prediction for {idx}...")
            user_id, movie_id = idx[0], idx[1]  # Extract userId and movieId from index

            # Check if user/movie exist in training matrix
            if user_id not in self.utility_matrix.index or movie_id not in self.utility_matrix.columns:
                skipped_count += 1
                continue  # Skip if not present in training

            # Calculate prediction for the specific (user, movie) pair
            try:
                predicted_rating_raw = self._predict_rating(user_id, movie_id)
                # Apply clipping here, before calculating metrics
                predicted_rating_clipped = np.clip(predicted_rating_raw, min_rating, max_rating)

                # Add values to lists
                true_ratings.append(true_rating)
                predicted_ratings.append(predicted_rating_clipped)

            except ValueError as e:
                skipped_count += 1
                continue
            except Exception as e:  # Catch other unexpected exceptions
                logging.info(f"Unexpected exception for ({user_id}, {movie_id}): {e}")
                skipped_count += 1
                continue

        end_time = time.time()

        logging.info(f"# Evaluation completed in {end_time - start_time:.2f} seconds.")
        if skipped_count > 0:
            logging.info(f"#   Skipped {skipped_count} ratings (user/movie not in training or error).")
        if not true_ratings:  # If there are no valid ratings to evaluate
            logging.info("#   No valid ratings found for evaluation.")
            return np.nan, np.nan

        # Calculate MAE and RMSE using collected lists
        mae = mean_absolute_error(true_ratings, predicted_ratings)  # MAE
        rmse = root_mean_squared_error(true_ratings, predicted_ratings)  # RMSE

        return mae, rmse
