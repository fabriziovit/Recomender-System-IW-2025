import time
from typing import Union
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
    """Distanza di Pearson (1 - |correlazione|) normalizzata tra 0 e 1"""
    corr, _ = pearsonr(x, y)
    if flag:
        return corr  # Restituisci la correlazione
    return 1 - np.abs(corr)  # Restituisci la distanza, che è 1 - |correlazione|


def hold_out_random(df_ratings: pd.DataFrame, valid_size=0.15, test_size=0.15, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1. Splitt in training e test dataframe
    train_ratings_df, val_test_ratings_df = train_test_split(df_ratings, test_size=valid_size + test_size, random_state=42)
    # 2. Splitto il training in training e validation
    valid_ratings_df, test_ratings_df = train_test_split(val_test_ratings_df, test_size=test_size, random_state=42)
    print(f"# Train Rows: {len(train_ratings_df)}, Valid Rows: {len(valid_ratings_df)}, Test Rows: {len(test_ratings_df)}")
    return train_ratings_df, valid_ratings_df, test_ratings_df


def get_train_valid_test_matrix(df_ratings: pd.DataFrame, all_movies_id: pd.Index, all_user_ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1. Split dei dati in training, validation e test da ratings
    df_train_ratings, df_valid_ratings, df_test_ratings = hold_out_random(df_ratings)
    # 2. Creo le matrici train, valid e test
    train_matrix: pd.DataFrame = df_train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    valid_matrix: pd.DataFrame = df_valid_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    test_matrix: pd.DataFrame = df_test_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Train-matrix: {train_matrix.shape}, Valid-matrix: {valid_matrix.shape}, Test-matrix: {test_matrix.shape}")
    # 3. Mi assicuro che le matrici abbiano tutte le colonne (movie_ids)
    train_matrix = train_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    valid_matrix = valid_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    test_matrix = test_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    # 4. Mi assicuro che le matrici abbiano tutte le righe (user_ids)
    train_matrix = train_matrix.reindex(index=all_user_ids, fill_value=0.0)
    valid_matrix = valid_matrix.reindex(index=all_user_ids, fill_value=0.0)
    test_matrix = test_matrix.reindex(index=all_user_ids, fill_value=0.0)
    print(f"# Train-matrix: {train_matrix.shape}, Valid-matrix: {valid_matrix.shape}, Test-matrix: {test_matrix.shape}")
    return train_matrix, valid_matrix, test_matrix


def min_max_normalize_mean(values: Union[int, float, pd.Series], min_val: float = 0.0, max_val: float = 5.0) -> float:
    """Calcola il rating medio normalizzato per un film"""
    if min_val == max_val:
        return 0.0  # Se max e min sono uguali, restituisco 0
    if isinstance(values, pd.Series):
        if values.empty:
            return 0.0  # Se non ci sono ratings, restituisce 0
        # Calcola il rating medio per il film
        avg_val = values.mean()
        return (avg_val - min_val) / (max_val - min_val)
    elif isinstance(values, (int, float)):  # Se è un singolo valore float
        return (values - min_val) / (max_val - min_val)
    else:
        raise ValueError("min_max_normalize_mean accetta solo int, float o pd.Series.")


def min_max_normalize_values(values: Union[int, float, pd.Series], min_val: float = 0.0, max_val: float = 5.0) -> float:
    """Normalizza i rating tra 0 e 1. Funziona sia per singoli valori float che per pd.Series, restituendo sempre un float."""
    if max_val == min_val:
        return 0.0  # Se max e min sono uguali, restituisco 0
    if isinstance(values, pd.Series):
        if values.empty:
            return 0.0  # Se non ci sono ratings, restituisco 0
        normalized_ratings = (values - min_val) / (max_val - min_val)
        return min_max_normalize_mean(normalized_ratings)  # Ritorna un singolo valore float
    elif isinstance(values, (int, float)):  # Se è un singolo valore float
        return (values - min_val) / (max_val - min_val)
    else:
        raise ValueError("min_max_normalize_values accetta solo int, float o pd.Series.")
