import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MF_SGD_User_Based:

    def __init__(self, num_factors, learning_rate, lambda_reg, n_epochs, utility_matrix, train_matrix, valid_matrix):
        """Initializes the Matrix Factorization model with SGD with bias."""
        self.num_factors: int = num_factors
        self.learning_rate: float = learning_rate
        self.lambda_reg: float = lambda_reg
        self.n_epochs: int = n_epochs
        self._is_fitted: bool = False

        self._utility_matrix: pd.DataFrame = utility_matrix

        self._train_samples: list[tuple] = None
        self._train_matrix: pd.DataFrame = train_matrix

        if not utility_matrix.shape[0] == train_matrix.shape[0]:
            raise ValueError("The utility and train matrices must have the same number of users (rows).")
        if not utility_matrix.shape[1] == train_matrix.shape[1]:
            raise ValueError("The utility and train matrices must have the same number of items (columns).")

        self._valid_samples: list[tuple] = None
        self._valid_matrix: pd.DataFrame = valid_matrix
        if not utility_matrix.shape[0] == train_matrix.shape[0]:
            raise ValueError("The utility and valid matrices must have the same number of users (rows).")
        if not utility_matrix.shape[1] == train_matrix.shape[1]:
            raise ValueError("The utility and valid matrices must have the same number of items (columns).")

        self.X_user_factors: np.ndarray = None  # Matrix X (User) Latent Factors
        self.Y_item_factors: np.ndarray = None  # Matrix Y (Item) Latent Factors

        self.user_biases: np.ndarray = None  # User bias
        self.item_biases: np.ndarray = None  # Item bias

        self.user_ids_map: dict = {}  # Map user_id -> matrix index
        self.item_ids_map: dict = {}  # Map movie_id -> matrix index

        self.train_mean: float = None  # Global mean on training set

    def _get_samples(self, matrix) -> list[tuple]:
        """Return a list of samples (user_index, item_index, rating) from a matrix."""
        logging.info("Creating samples...")
        samples = []
        stacked_matrix = matrix.stack()  # Reshape DataFrame to Series (multi-index)

        # Filter for ratings > 0.0 directly on the Series
        explicit_ratings_series = stacked_matrix[stacked_matrix > 0.0]

        for index, rating in explicit_ratings_series.items():  # Iterate over the filtered Series
            user_id_idx = self.user_ids_map.get(index[0])  # Get user_index from user_id (using map)
            item_id_idx = self.item_ids_map.get(index[1])  # Get item_index from item_id (using map)
            samples.append((user_id_idx, item_id_idx, rating))  # Append sample
        return samples

    def _get_mse_loss_regularized(self, samples) -> float:
        """Calculates the loss function (MSE with L2 regularization) including bias and global mean."""
        user_indices, item_indices, real_ratings = zip(*samples)

        user_indices = np.array(user_indices)
        item_indices = np.array(item_indices)
        real_ratings = np.array(real_ratings)

        predicted_ratings = (
            self.train_mean
            + self.user_biases[user_indices]
            + self.item_biases[item_indices]
            + np.sum(self.X_user_factors[user_indices] * self.Y_item_factors[item_indices], axis=1)
        )

        partial_res = 0.5 * self.lambda_reg * (np.sum(self.X_user_factors**2) + np.sum(self.Y_item_factors**2) + np.sum(self.user_biases**2) + np.sum(self.item_biases**2))
        errors = real_ratings - predicted_ratings
        mse_loss = 0.5 * np.sum(errors**2)
        return mse_loss + partial_res

    def _predict_rating(self, user_index: int, item_index: int) -> float:
        """Predicts the rating with bias and global mean (de-normalization)."""
        return self.train_mean + self.user_biases[user_index] + self.item_biases[item_index] + np.dot(self.X_user_factors[user_index], self.Y_item_factors[item_index])

    def _get_predictions(self, user_id: int, matrix: pd.DataFrame, exclude: bool = True) -> pd.DataFrame:
        """Returns predictions for a specific user."""
        if user_id not in self.user_ids_map:
            raise ValueError(f"User ID {user_id} not found in training set.")
        predicted_ratings_list = []
        item_ids = matrix.columns  # All movie ids from the matrix
        user_index = self.user_ids_map[user_id]
        seen_movies_ids = matrix.loc[user_id][matrix.loc[user_id] > 0].index.tolist()  # Movies seen by the user
        for item_id in item_ids:
            if exclude and item_id in seen_movies_ids:
                continue  # Skip movies already seen if exclude is True
            item_index = self.item_ids_map[item_id]
            predicted_rating = self._predict_rating(user_index, item_index)
            predicted_ratings_list.append({"movieId": item_id, "predicted_rating": predicted_rating})
        predictions_df = pd.DataFrame(predicted_ratings_list).set_index("movieId")
        predictions_df.columns = ["values"]
        return predictions_df.sort_values(by="values", ascending=False)

    def _stochastic_gradient_descent(self, num_factors: int, learning_rate: float, lambda_term: float, refit: bool = False, evaluation_output: list = None) -> None:
        if not self._is_fitted or refit:
            logging.info("Performing training with bias and global mean...")
            logging.info(f"Hyperparameters: num_factors={num_factors}, learning_rate={learning_rate}, lambda={lambda_term}")  # Log hyperparameters
            best_valid_loss_reg = float("inf")
            best_epoch = 0
            patience = 10
            epochs_no_improve = 0

            best_user_factors = None
            best_item_factors = None
            best_user_biases = None
            best_item_biases = None

            for epoch in range(0, self.n_epochs):
                loss_mse = 0
                np.random.shuffle(self._train_samples)
                for user_index, item_index, real_rating in self._train_samples:
                    # 0. Calculate error
                    predicted_rating = self._predict_rating(user_index, item_index)
                    error_ij = real_rating - predicted_rating
                    # 1. Retrieve latent factors and biases
                    user_factor = self.X_user_factors[user_index]  # User latent factors
                    item_factor = self.Y_item_factors[item_index]  # Item latent factors
                    user_bias = self.user_biases[user_index]  # User bias
                    item_bias = self.item_biases[item_index]  # Item bias
                    # 2. Update latent factors and biases with gradient calculation
                    user_gradient = -(error_ij * item_factor) + (lambda_term * user_factor)  # Gradient for user factors
                    item_gradient = -(error_ij * user_factor) + (lambda_term * item_factor)  # Gradient for item factors
                    user_bias_gradient = -error_ij + (lambda_term * user_bias)  # Gradient for user bias
                    item_bias_gradient = -error_ij + (lambda_term * item_bias)  # Gradient for item bias
                    # 4. Update latent factors and biases
                    self.X_user_factors[user_index] -= self.learning_rate * user_gradient  # Update user factors
                    self.Y_item_factors[item_index] -= self.learning_rate * item_gradient  # Update item factors
                    self.user_biases[user_index] -= self.learning_rate * user_bias_gradient  # Update user bias
                    self.item_biases[item_index] -= self.learning_rate * item_bias_gradient  # Update item bias
                    # 5. Sum of squared error
                    loss_mse += error_ij**2
                # 6. Calculate mean squared error (MSE)
                loss_mse /= len(self._train_samples)
                # 7. Calculate regularized mean squared error (MSE + Weight Decay (L2))
                train_loss_reg = self._get_mse_loss_regularized(self._train_samples)
                valid_loss_reg = self._get_mse_loss_regularized(self._valid_samples)

                log_msg = f"Epoch {epoch+1}/{self.n_epochs}, Train-Loss: {loss_mse:.10f}, Train-Loss-Reg: {train_loss_reg:.10f}, Valid-Loss-Reg: {valid_loss_reg:.10f}, K: {num_factors}, lrate: {learning_rate}, lambda: {lambda_term}"
                logging.info(log_msg)
                if evaluation_output is not None:
                    evaluation_output.append(log_msg)

                # Early stopping criteria
                if valid_loss_reg < best_valid_loss_reg:
                    best_valid_loss_reg = valid_loss_reg
                    best_epoch = epoch + 1
                    epochs_no_improve = 0

                    best_user_factors = self.X_user_factors.copy()
                    best_item_factors = self.Y_item_factors.copy()

                    best_user_biases = self.user_biases.copy()
                    best_item_biases = self.item_biases.copy()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logging.info(f"Early stopping activated at epoch {epoch+1}!")
                        logging.info(f"Valid-Loss-Reg not improved for {patience} epochs.")
                        log_msg = f"Best epoch: {best_epoch} with Train-Loss: {loss_mse:.10f}, Train-Loss-Reg:{train_loss_reg:.10f},  Valid-Loss-Reg: {best_valid_loss_reg:.10f}"
                        logging.info(log_msg)
                        if evaluation_output is not None:
                            evaluation_output.append(log_msg)
                        # Restore best parameters
                        self.X_user_factors = best_user_factors
                        self.Y_item_factors = best_item_factors
                        self.user_biases = best_user_biases
                        self.item_biases = best_item_biases
                        return

    def fit(self, refit: bool = False, evaluation_output: list = None) -> None:

        user_ids = self._train_matrix.index
        item_ids = self._train_matrix.columns

        # Mapping user_id and item_id to matrix indices
        self.user_ids_map = {user_id: index for index, user_id in enumerate(user_ids)}
        self.item_ids_map = {item_id: index for index, item_id in enumerate(item_ids)}

        # Get Samples for train and test
        self._train_samples = self._get_samples(self._train_matrix)
        self._valid_samples = self._get_samples(self._valid_matrix)
        logging.info(f"train_samples: {len(self._train_samples)}, valid_samples: {len(self._valid_samples)}")

        # Initialize latent factors and biases
        self.X_user_factors = np.random.normal(0, 0.01, (len(user_ids), self.num_factors))
        self.Y_item_factors = np.random.normal(0, 0.01, (len(item_ids), self.num_factors))
        self.user_biases = np.zeros(len(user_ids), dtype=np.float64)
        self.item_biases = np.zeros(len(item_ids), dtype=np.float64)

        # Calculate global mean on training set for normalization
        self.train_mean = self._train_matrix[self._train_matrix > 0.0].stack().mean()
        logging.info(f"Global mean on training set: {self.train_mean:.10f}")

        # Train the model
        self._stochastic_gradient_descent(self.num_factors, self.learning_rate, self.lambda_reg, refit, evaluation_output)
        self._is_fitted = True

    def get_recommendations(self, utility_matrix: pd.DataFrame, user_id: pd.Index) -> pd.DataFrame:
        """Returns predictions for the specified user."""
        # Exclude movies already seen by the user
        recomm_df = self._get_predictions(user_id, utility_matrix, exclude=True)
        return recomm_df

    @classmethod
    def load_model(cls, filepath):
        """Loads a previously saved model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model

    def save_model(self, filepath):
        """Saves the trained model to disk using pickle."""
        if not self._is_fitted:
            raise ValueError("The model must be trained (fit) before it can be saved.")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logging.info(f"Model successfully saved to: {filepath}")

    # **************************************************************************** #

    def evaluate_mea_rmse(self, test_matrix: pd.DataFrame, min_rating: float = 0.5, max_rating: float = 5.0):
        """Evaluates the model on the test set calculating MAE and RMSE."""
        if not self._is_fitted:
            raise RuntimeError("The MF model must be trained before evaluation. Call fit().")

        logging.info("Starting MAE/RMSE evaluation...")

        true_ratings = []
        predicted_ratings = []
        skipped_count = 0
        processed_count = 0
        start_time = time.time()

        # Efficient iteration on Test Set using stack()
        test_ratings_series = test_matrix[test_matrix > 0].stack()
        total_ratings_to_predict = len(test_ratings_series)

        if total_ratings_to_predict == 0:
            logging.warning("No rating > 0 found in the provided test_matrix for evaluation.")
            return np.nan, np.nan

        logging.info(f"Evaluating on {total_ratings_to_predict} ratings in test set...")

        # Iterate on the resulting Series (multi-level index user_id, movie_id)
        for idx, true_rating in test_ratings_series.items():
            user_id, movie_id = idx  # Extract userId and movieId from index

            processed_count += 1

            # Get internal indices corresponding to IDs
            user_idx = self.user_ids_map.get(user_id)
            item_idx = self.item_ids_map.get(movie_id)

            # Skip if user or item were not present in training set
            # The model cannot make predictions for data never seen during training
            if user_idx is None or item_idx is None:
                skipped_count += 1
                continue

            # Calculate prediction using internal indices
            try:
                predicted_rating_raw = self._predict_rating(user_idx, item_idx)

                # Apply clipping to bring prediction back to valid range
                predicted_rating_clipped = np.clip(predicted_rating_raw, min_rating, max_rating)

                # Add values to lists
                true_ratings.append(true_rating)
                predicted_ratings.append(predicted_rating_clipped)

            except IndexError:
                # This could happen if maps or factors are inconsistent
                logging.error(f"Index out of bounds for ({user_id},{movie_id}) -> ({user_idx},{item_idx}). Skipping.")
                skipped_count += 1
                continue
            except Exception as e:
                logging.error(f"Generic error during prediction for ({user_id},{movie_id}): {e}")
                skipped_count += 1
                continue

        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f"MAE/RMSE evaluation completed in {elapsed_time:.2f} seconds.")
        if skipped_count > 0:
            logging.info(f"  Skipped {skipped_count} ratings (user/item not in training or error).")

        # Calculate MAE and RMSE if there were valid predictions
        if not true_ratings:
            logging.warning("No valid predictions generated for MAE/RMSE calculation.")
            return np.nan, np.nan
        else:
            mae = mean_absolute_error(true_ratings, predicted_ratings)
            rmse = root_mean_squared_error(true_ratings, predicted_ratings)  # squared=False for RMSE
            logging.info(f"  MAE:  {mae:.10f}")
            logging.info(f"  RMSE: {rmse:.10f}")
            return mae, rmse
