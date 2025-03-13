import time
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split


def load_movielens_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_movies = pd.read_csv(path + "movies.csv", index_col="movieId", on_bad_lines="warn")
    df_ratings = pd.read_csv(path + "ratings.csv", on_bad_lines="warn")
    df_tags = pd.read_csv(path + "tags.csv", on_bad_lines="warn")
    return df_movies, df_ratings, df_tags


def pearson_distance(x, y, flag: bool = False):
    """Distanza di Pearson (1 - |correlazione|)"""
    corr, _ = pearsonr(x, y)
    if flag:
        return corr  # Restituisci la correlazione
    return 1 - np.abs(corr)  # Restituisci la distanza, che è 1 - |correlazione|


def hold_out_random(df_ratings: pd.DataFrame, valid_size=0.15, test_size=0.15, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1. Splitto in training e test
    train_ratings_df, val_test_ratings_df = train_test_split(df_ratings, test_size=valid_size + test_size, random_state=42)
    # 2. Splitto il training in training e validation
    valid_ratings_df, test_ratings_df = train_test_split(val_test_ratings_df, test_size=test_size, random_state=42)
    print(f"# Train Rows: {len(train_ratings_df)}, Valid Rows: {len(valid_ratings_df)}, Test Rows: {len(test_ratings_df)}")
    return train_ratings_df, valid_ratings_df, test_ratings_df


def get_train_valid_test_matrix(df_ratings: pd.DataFrame, all_movies_id: pd.Index, all_user_ids: pd.Index):

    # 1. Split dei dati in training, validation e test da ratings
    df_train_ratings, df_valid_ratings, df_test_ratings = hold_out_random(df_ratings)

    train_matrix: pd.DataFrame = df_train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    valid_matrix: pd.DataFrame = df_valid_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    test_matrix: pd.DataFrame = df_test_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Train-matrix: {train_matrix.shape}, Valid-matrix: {valid_matrix.shape}, Test-matrix: {test_matrix.shape}")

    # 2. Mi assicuro che le matrici abbiano tutte le colonne (movie_ids)
    train_matrix = train_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    valid_matrix = valid_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    test_matrix = test_matrix.reindex(columns=all_movies_id, fill_value=0.0)

    # 3. Mi assicuro che le matrici abbiano tutte le righe (user_ids)
    train_matrix = train_matrix.reindex(index=all_user_ids, fill_value=0.0)
    valid_matrix = valid_matrix.reindex(index=all_user_ids, fill_value=0.0)
    test_matrix = test_matrix.reindex(index=all_user_ids, fill_value=0.0)
    print(f"# Train-matrix: {train_matrix.shape}, Valid-matrix: {valid_matrix.shape}, Test-matrix: {test_matrix.shape}")
    return train_matrix, valid_matrix, test_matrix


def compute_mean_form_movie(movie_id: str, df_ratings: pd.DataFrame) -> float:
    """Calcola il rating medio normalizzato per un film"""
    min_rating_value: float = 0.0
    max_rating_value: float = 5.0
    movie_ratings = df_ratings[df_ratings["movieId"] == movie_id]["rating"]
    if movie_ratings.empty:
        return 0.0  # Se non ci sono rating, restituisce 0
    # Calcola il rating medio per il film
    avg_rating = movie_ratings.mean()
    # Normalizza il rating medio [0,5] -> [0,1]
    return (avg_rating - min_rating_value) / (max_rating_value - min_rating_value)


def compute_hybrid_reward(similarity: float, mean_reward: float, beta: float = 0.5) -> float:
    # Calcola della reward: combinazione lineare di similarità e rating medio del film selezionato
    return beta * similarity + (1 - beta) * mean_reward


# def pearson_distance_manual(u: np.ndarray, v: np.ndarray, flag: bool = False) -> float:
#     """Distanza di Pearson (1 - |correlazione|)"""
#     u_mean = np.mean(u)
#     v_mean = np.mean(v)
#     num = np.sum((u - u_mean) * (v - v_mean))  # Numeratore
#     den = np.sqrt(np.sum((u - u_mean) ** 2)) * np.sqrt(np.sum((v - v_mean) ** 2))  # Denominatore
#     correlation = num / den if den != 0 else 0  # Correlazione di Pearson
#     if flag:
#         return correlation  # Restituisci la correlazione
#     return 1 - abs(correlation)  # Restituisci la distanza, che è 1 - |correlazione|
