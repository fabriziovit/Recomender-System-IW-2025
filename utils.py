import logging
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split


def load_movielens_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_movies = pd.read_csv(path + "movies.csv", index_col="movieId", on_bad_lines="warn")
    df_ratings = pd.read_csv(path + "ratings.csv", on_bad_lines="warn")
    df_tags = pd.read_csv(path + "tags.csv", on_bad_lines="warn")
    return df_movies, df_ratings, df_tags


def _hold_out_random_train_valid_test(
    df_ratings: pd.DataFrame,
    valid_size=0.15,  # Proportion of the original dataset for validation
    test_size=0.15,  # Proportion of the original dataset for test
    random_state=42,
    ret_valid: bool = False,
) -> Union[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:

    if not ret_valid:
        # If validation set is not wanted, test_size is the size of the test set.
        train_ratings_df, test_ratings_df = train_test_split(df_ratings, test_size=test_size, random_state=random_state)
        logging.info(f"# Train Rows: {len(train_ratings_df)}, Test Rows: {len(test_ratings_df)}")
        return train_ratings_df, test_ratings_df
    else:
        # Calculate the combined size of validation and test
        hold_out_size = valid_size + test_size
        if hold_out_size >= 1.0:
            raise ValueError("The sum of valid_size and test_size must be less than 1.")

        # First split: training vs. (validation + test)
        train_ratings_df, temp_hold_out_df = train_test_split(df_ratings, test_size=hold_out_size, random_state=random_state)

        # Calculate the proportion of the test set relative to the (validation + test) block
        # Example: if valid_size=0.15, test_size=0.15, then hold_out_size=0.30.
        # We want test_size to be 0.15 of the original, so 0.15/0.30 = 0.5 of the temp_hold_out_df block

        relative_test_size = test_size / hold_out_size
        valid_ratings_df, test_ratings_df = train_test_split(temp_hold_out_df, test_size=relative_test_size, random_state=random_state)

        logging.info(f"# Train Rows: {len(train_ratings_df)}, Valid Rows: {len(valid_ratings_df)}, Test Rows: {len(test_ratings_df)}")
        return train_ratings_df, valid_ratings_df, test_ratings_df


def _get_reindexd_matrix(matrix: pd.DataFrame, all_movies_id: pd.Index, all_user_ids: pd.Index) -> pd.DataFrame:
    matrix = matrix.reindex(columns=all_movies_id, fill_value=0.0)
    matrix = matrix.reindex(index=all_user_ids, fill_value=0.0)
    return matrix


def get_train_valid_test_matrix(
    df_ratings: pd.DataFrame,
    all_movies_id: pd.Index,
    all_user_ids: pd.Index,
    ret_valid: bool = False,
) -> Union[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:

    if not ret_valid:
        # Split data into training and test from ratings
        df_train_ratings, df_test_ratings = _hold_out_random_train_valid_test(df_ratings, ret_valid=False)
        # Create train and test matrices
        train_matrix: pd.DataFrame = df_train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        test_matrix: pd.DataFrame = df_test_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    else:
        # Split data into training, validation and test from ratings
        df_train_ratings, df_valid_ratings, df_test_ratings = _hold_out_random_train_valid_test(df_ratings, ret_valid=True)
        # Create train, valid and test matrices
        train_matrix: pd.DataFrame = df_train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        valid_matrix: pd.DataFrame = df_valid_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        test_matrix: pd.DataFrame = df_test_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Ensure that the matrices have all columns (movie_ids) and all rows (user_ids)
    train_matrix: pd.DataFrame = _get_reindexd_matrix(train_matrix, all_movies_id, all_user_ids)
    test_matrix: pd.DataFrame = _get_reindexd_matrix(test_matrix, all_movies_id, all_user_ids)

    if not ret_valid:
        logging.info(f"# Train-matrix: {train_matrix.shape}, Test-matrix: {test_matrix.shape}")
        return train_matrix, test_matrix
    else:
        valid_matrix = _get_reindexd_matrix(valid_matrix, all_movies_id, all_user_ids)
        logging.info(f"# Train-matrix: {train_matrix.shape}, Valid-matrix: {valid_matrix.shape}, Test-matrix: {test_matrix.shape}")
        return train_matrix, valid_matrix, test_matrix


def min_max_normalize(values: Union[int, float, pd.Series], min_val: float = None, max_val: float = None) -> float:
    """Normalizes values between 0 and 1"""
    if min_val is None or max_val is None:
        raise ValueError("min_max_normalize requires min_val and max_val.")
    if min_val == max_val:
        return 0.0  # If max and min are equal, return 0
    if isinstance(values, pd.Series):
        if values.empty:
            return 0.0  # If there are no ratings, return 0
        # If there are multiple values, calculate the mean
        avg_val = values.mean()
        return (avg_val - min_val) / (max_val - min_val)
    elif isinstance(values, (int, float)):  # If it's a single float value
        return (values - min_val) / (max_val - min_val)
    else:
        raise ValueError("min_max_normalize_mean accepts only int, float or pd.Series.")


def linear_epsilon_decay(initial_epsilon: float, num_round: int, decay: float = 0.00009) -> float:
    return initial_epsilon - (num_round * decay)  # linear


def log_epsilon_decay(initial_epsilon: float, num_round: int, decay_strength: float = 0.5) -> float:
    return initial_epsilon / (1 + decay_strength * np.log(num_round + 1))  # logarithmic


def exp_epsilon_decay(initial_epsilon: float, num_round: int, decay: float = 0.9995) -> float:
    return initial_epsilon * (decay**num_round)  # exponential
