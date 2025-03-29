import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances


def load_movielens_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_movies = pd.read_csv(path + "movies.csv", index_col="movieId", on_bad_lines="warn")
    df_ratings = pd.read_csv(path + "ratings.csv", on_bad_lines="warn")
    df_tags = pd.read_csv(path + "tags.csv", on_bad_lines="warn")
    return df_movies, df_ratings, df_tags


def pearson_similarity(x, y):
    return pearsonr(x, y)[0]


def pearson_distance(x, y):
    """Distanza di Pearson (1 - correlazione) normalizzata tra 0 e 1"""
    return 1 - pearson_similarity(x, y)  # Restituisci la distanza, che è 1 - correlazione


def compute_user_similarity_matrix(matrix) -> pd.DataFrame:
    """Calcola la matrice di similarità utente-utente usando pairwise_distances."""
    # pairwise_distances è 1 - pearson_correlation
    # Quindi 1 - (1 - pearson_correlation) = pearson_correlation
    similarity_matrix = 1 - pairwise_distances(matrix, metric="correlation", n_jobs=-1)
    similarity_df = pd.DataFrame(similarity_matrix, index=matrix.index, columns=matrix.index)
    return similarity_df


def _hold_out_random_train_valid_test(
    df_ratings: pd.DataFrame,
    valid_size=0.15,
    test_size=0.15,
    random_state=42,
    ret_valid: bool = False,
) -> Union[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    # Split in training e test dataframe
    train_ratings_df, test_ratings_df = train_test_split(df_ratings, test_size=valid_size + test_size, random_state=42)
    if not ret_valid:
        print(f"# Train Rows: {len(train_ratings_df)}, Test Rows: {len(test_ratings_df)}")
        return train_ratings_df, test_ratings_df
    else:
        # Split il training in training effettivo e validation
        valid_ratings_df, test_ratings_df = train_test_split(test_ratings_df, test_size=test_size, random_state=42)
        print(f"# Train Rows: {len(train_ratings_df)}, Valid Rows: {len(valid_ratings_df)}, Test Rows: {len(test_ratings_df)}")
    return train_ratings_df, valid_ratings_df, test_ratings_df


def _get_reindexd_matrix(matrix: pd.DataFrame, all_movies_id: pd.Index, all_user_ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        # Split dei dati in training e test da ratings
        df_train_ratings, df_test_ratings = _hold_out_random_train_valid_test(df_ratings, ret_valid=False)
        # Creo le matrici train e test
        train_matrix: pd.DataFrame = df_train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        test_matrix: pd.DataFrame = df_test_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    else:
        # Split dei dati in training, validation e test da ratings
        df_train_ratings, df_valid_ratings, df_test_ratings = _hold_out_random_train_valid_test(df_ratings, ret_valid=True)
        # Creo le matrici train, valid e test
        train_matrix: pd.DataFrame = df_train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        valid_matrix: pd.DataFrame = df_valid_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        test_matrix: pd.DataFrame = df_test_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Mi assicuro che le matrici abbiano tutte le colonne (movie_ids) e tutte le righge (user_ids)
    train_matrix = _get_reindexd_matrix(train_matrix, all_movies_id, all_user_ids)
    test_matrix = _get_reindexd_matrix(test_matrix, all_movies_id, all_user_ids)

    if not ret_valid:
        print(f"# Train-matrix: {train_matrix.shape}, Test-matrix: {test_matrix.shape}")
        return train_matrix, test_matrix
    else:
        valid_matrix = _get_reindexd_matrix(valid_matrix, all_movies_id, all_user_ids)
        print(f"# Train-matrix: {train_matrix.shape}, Valid-matrix: {valid_matrix.shape}, Test-matrix: {test_matrix.shape}")
        return train_matrix, valid_matrix, test_matrix


def min_max_normalize(values: Union[int, float, pd.Series], min_val: float = None, max_val: float = None) -> float:
    """Normalizza i valori tra 0 e 1"""
    if min_val is None or max_val is None:
        raise ValueError("min_max_normalize richiede min_val e max_val.")
    if min_val == max_val:
        return 0.0  # Se max e min sono uguali, restituisco 0
    if isinstance(values, pd.Series):
        if values.empty:
            return 0.0  # Se non ci sono ratings, restituisce 0
        # Se ci sono più valori, calcola la media
        avg_val = values.mean()
        return (avg_val - min_val) / (max_val - min_val)
    elif isinstance(values, (int, float)):  # Se è un singolo valore float
        return (values - min_val) / (max_val - min_val)
    else:
        raise ValueError("min_max_normalize_mean accetta solo int, float o pd.Series.")


def linear_epsilon_decay(initial_epsilon: float, num_round: int, decay: float = 0.00005) -> float:
    return initial_epsilon - (num_round * decay)  # lineare


def log_epsilon_decay(initial_epsilon: float, num_round: int, decay_strength: float = 1.0) -> float:
    return initial_epsilon / (1 + decay_strength * np.log(num_round + 1))  # logaritmico


def exp_epsilon_decay(initial_epsilon: float, num_round: int, decay: float = 0.9995) -> float:
    return initial_epsilon * (decay**num_round)  # esponenziale
