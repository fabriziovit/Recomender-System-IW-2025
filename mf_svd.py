import datetime
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import load_movielens_data, hold_out_random_train_test, pearson_distance


def precompute_user_means(users_x_movie: pd.DataFrame) -> dict:
    """Calcola la media dei rating per ogni utente ignorando gli 0.
    Restituisce un dizionario in cui la chiave è l'user_id e il valore è la media."""
    # Sostituisci gli 0 con NaN, calcola la media lungo le colonne e sostituisci eventuali NaN con 0.
    user_means = users_x_movie.replace(0, np.nan).mean(axis=1).fillna(0).to_dict()
    return user_means


def predict_rating_mean_np_migliorato(movie_id: int, user_id: int, similiars_x_movie_s: pd.Series, dist_s: pd.Series, user_means: dict) -> float:
    """Calcola la media pesata ignorando rating mancanti (0.0) in maniera vettoriale.
    Non sostituisce i rating 0 con le medie degli utenti."""
    # Converti i rating dei simili e le similarità in array NumPy
    ratings = similiars_x_movie_s.to_numpy()
    sim_uv = np.maximum(0, 1 - dist_s.to_numpy())

    # Identifica gli indici dei rating non nulli
    non_zero_rating_indices = ratings != 0

    # Applica il filtro ai rating e alle similarità
    filtered_ratings = ratings[non_zero_rating_indices]
    filtered_sim_uv = sim_uv[non_zero_rating_indices]

    # Calcola numeratore e denominatore solo per i rating non nulli
    numeratore = np.dot(filtered_sim_uv, filtered_ratings)
    denominatore = np.sum(np.abs(filtered_sim_uv))

    return numeratore / denominatore if denominatore != 0 else 0


def predict_rating_mean_np(movie_id: int, user_id: int, similiars_x_movie_s: pd.Series, dist_s: pd.Series, user_means: dict) -> float:
    """Calcola la media pesata NON ignorando rating mancanti (0.0) in maniera vettoriale,
    utilizzando le medie pre-calcolate per ogni utente.
    Se il rating di un utente simile è 0, viene sostituito con la sua media pre-calcolata."""
    # Converti i rating dei simili in un array NumPy
    ratings = similiars_x_movie_s.to_numpy()

    # Estrai le medie degli utenti simili dal dizionario
    means = np.array([user_means[u] for u in similiars_x_movie_s.index])

    # Sostituisci i rating a 0 con la media pre-calcolata dell'utente simile
    adjusted_ratings = np.where(ratings == 0, means, ratings)

    # Calcola il vettore delle similarità: max(0, 1 - distanza)
    sim_uv = np.maximum(0, 1 - dist_s.to_numpy())

    # Calcola numeratore e denominatore in modo vettoriale
    numeratore = np.dot(sim_uv, adjusted_ratings)
    denominatore = np.sum(np.abs(sim_uv))

    return numeratore / denominatore if denominatore != 0 else 0


def get_train_and_test_matrix(df_ratings: pd.DataFrame, all_movies_id: pd.Index):
    # Split dei dati in training e test
    df_train_ratings, df_test_ratings = hold_out_random_train_test(df_ratings)

    train_ratings_matrix = df_train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    test_ratings_matrix = df_test_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Train ratings-matrix: {train_ratings_matrix.shape}, Test ratings-matrix: {test_ratings_matrix.shape}")

    # Assicuriamoci che entrambe le matrici abbiano tutte le colonne
    train_ratings_matrix = train_ratings_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    test_ratings_matrix = test_ratings_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    print(f"# Train ratings-matrix: {train_ratings_matrix.shape}, Test ratings-matrix: {test_ratings_matrix.shape}")

    return train_ratings_matrix, test_ratings_matrix


class CF_SVD_DimensionalityReduction_UserBased:

    def __init__(self, model_user: NearestNeighbors):
        """Inizializza il Recommender con un modello di machine learning (NearestNeighbors)."""
        if not isinstance(model_user, NearestNeighbors):
            raise ValueError("model_user must be NearestNeighbors")
        self.model_user = model_user
        self._is_fitted_on_matrix_user = False

    def fit_user_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """Addestra il modello user-based (NearestNeighbors) su matrix se necessario o se forzato."""
        if not self._is_fitted_on_matrix_user or re_fit:
            self.model_user.fit(matrix)
            self._is_fitted_on_matrix_user = True
            print("*** Modello model_user addestrato su matrix. ***")
        else:
            print("*** Modello model_user già addestrato su matrix. Salto l'addestramento. ***")

    def get_predictions(self, user_id: int, factored_matrix: pd.DataFrame, train_matrix: pd.DataFrame, user_means: dict) -> pd.DataFrame:
        """Raccomanda film a un utente dato utilizzando il filtraggio collaborativo user-based."""
        all_users = train_matrix.index

        # Seleziona le caratteristiche dell'utente specificato dalla matrice utenti-film
        user_index = train_matrix.index.get_loc(user_id)
        user_features = factored_matrix[user_index].reshape(1, -1)

        # Trova gli utenti simili usando NearestNeighbors
        distances, pos_indexes = self.model_user.kneighbors(user_features)

        # Esclude l'utente target (la prima posizione)
        similar_users = [all_users[pos_idx] for pos_idx in pos_indexes.squeeze().tolist()[1:]]
        dist = pd.Series(distances.squeeze().tolist()[1:], index=similar_users)
        print(f"#Utenti simili per user {user_id}: {similar_users} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in dist.items()]}")

        # Recupera le valutazioni degli utenti simili
        users_x_movies_similar_df = train_matrix.loc[similar_users]

        # Rimuove i film già visti dall'utente target dalla lista dei raccomandabili
        seen_mask = train_matrix.loc[user_id] > 0
        seen_movies = train_matrix.loc[user_id][seen_mask].index
        users_x_movies_similar_df = users_x_movies_similar_df.loc[:, ~users_x_movies_similar_df.columns.isin(seen_movies)]

        # I film raccomandabili sono le colonne rimaste
        res_movies_ids = users_x_movies_similar_df.columns

        # Per ogni film, predice il rating basandosi sui rating degli utenti simili
        values = [predict_rating_mean_np(mid, user_id, users_x_movies_similar_df[mid], dist, user_means) for mid in res_movies_ids]

        return pd.DataFrame(values, index=res_movies_ids, columns=["values"]).sort_values(by="values", ascending=False)


def precompute_all_predictions(
    recomm: CF_SVD_DimensionalityReduction_UserBased,
    train_ratings_matrix: pd.DataFrame,
    utility_matrix: pd.DataFrame,
    common_users: set,
    user_means: dict,
) -> dict:
    """Calcola e salva le predizioni per tutti gli utenti in common_users.
    Restituisce un dizionario: {user_id: predictions_df}"""
    all_user_predictions = {}
    for user_id in common_users:
        all_user_predictions[user_id] = recomm.get_predictions(user_id, train_ratings_matrix, utility_matrix, user_means)
    return all_user_predictions


def main():
    # Carica il dataset MovieLens
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")

    all_movies_id = df_movies.index

    # Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Numero di utenti totali: {len(utility_matrix.index)}")
    print(f"# Numero di movies totali: {len(utility_matrix.columns)}")
    print(f"# Utility ratings-matrix: {utility_matrix.shape}")

    # Precomputazione della media per ogni utente
    user_means = precompute_user_means(utility_matrix)

    # Splitting in training e test matrix
    train_matrix, test_matrix = get_train_and_test_matrix(df_ratings, all_movies_id)

    best_rmse = float("inf")
    best_mae = float("inf")
    best_params = None
    results = {}
    n_components_list = [10]
    n_iter_list = [30]

    for n_components in n_components_list:
        for n_iter in n_iter_list:
            # Modello SVD per la matrix-factorization
            svd = TruncatedSVD(n_components, n_iter)
            #! Non viene fatta l'imputazione dei rating mancanti
            # Fit del modello SVD on Training
            factored_train_matrix = svd.fit_transform(train_matrix)

            # Parametri di test
            K_list = [5]
            nn_list = [5, 10]
            relevant_value = 3.0

            results_all = {}

            # Determina gli utenti comuni (che useranno le predizioni precompute)
            common_users = set(train_matrix.index) & set(test_matrix.index)
            print(f"# Numero di utenti comuni: {len(common_users)}")

            for NN in nn_list:
                knn_model = NearestNeighbors(n_neighbors=NN + 1, metric=pearson_distance, algorithm="brute", n_jobs=-1)
                recomm = CF_SVD_DimensionalityReduction_UserBased(knn_model)
                recomm.fit_user_model(factored_train_matrix, re_fit=True)

                # Precomputo le predizioni per tutti gli utenti comuni
                precomputed_predictions = precompute_all_predictions(recomm, factored_train_matrix, train_matrix, common_users, user_means)


if __name__ == "__main__":
    main()
