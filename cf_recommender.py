import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import load_movielens_data, pearson_distance


class CollaborativeRecommender:

    def __init__(self, model_item: NearestNeighbors, model_user: NearestNeighbors):
        """
        Inizializza il Recommender con un modello di machine learning.
        """
        if not isinstance(model_item, NearestNeighbors):
            raise ValueError(f"model_item must be NearestNeighbors")
        if not model_item is None:
            self.model_item = model_item
        if not isinstance(model_user, NearestNeighbors):
            raise ValueError(f"model_user must be NearestNeighbors")
        if not model_user is None:
            self.model_user = model_user

        self._transposed_matrix = None
        self._is_fitted_on_matrix_T_item = False
        self._is_fitted_on_matrix_user = False

    def fit_item_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """
        Addestra il modello item-based su matrix.T se necessario o se forzato.
        """
        if not self._is_fitted_on_matrix_T_item or re_fit:
            self._transposed_matrix = matrix.T
            self.model_item.fit(self._transposed_matrix)
            self._is_fitted_on_matrix_T_item = True
            print("# Modello model_item addestrato su matrix.T (movies x users).")
        else:
            print("# Modello model_item già addestrato su matrix.T (movies x users): Salto l'addestramento.")

    def fit_user_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """
        Addestra il modello user-based su matrix se necessario o se forzato.
        """
        if not self._is_fitted_on_matrix_user or re_fit:
            self.model_user.fit(matrix)
            self._is_fitted_on_matrix_user = True
            print("# Modello model_user addestrato su matrix (users x movies).")
        else:
            print("# Modello model_user già addestrato su matrix (users x movies): Salto l'addestramento.")

    def get_item_recommendations(self, movie_id: int, matrix: pd.DataFrame, df_movies: pd.DataFrame) -> pd.DataFrame:
        """
        Raccomanda film simili a un film dato utilizzando il filtraggio collaborativo item-based.
        """

        if movie_id not in self._transposed_matrix.index:
            print(f"Movie ID {movie_id} non trovato nella matrice.")
            raise ValueError(f"Movie ID {movie_id} non trovato nella matrice.")

        all_movie_ids = self._transposed_matrix.index

        # 1. Seleziona le caratteristiche del film specificato dalla matrice trasposta (movieId x userId)
        movie_features = self._transposed_matrix.loc[movie_id].values.reshape(1, -1)

        # 2. Trova i film più simili usando il modello NearestNeighbors
        distances, pos_indexes = self.model_item.kneighbors(movie_features)

        # Ricava gli ID dei film simili utilizzando gli indici posizionali
        similar_movies_list = [all_movie_ids[pos_idx] for pos_idx in pos_indexes.squeeze().tolist()[1:]]
        dist = pd.Series(distances.squeeze().tolist()[1:], index=similar_movies_list)

        print(f"Film simili trovati per il film: {df_movies.loc[movie_id, 'title']} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in dist.items()]}")

        # Crea un DataFrame con le raccomandazioni dei film
        return df_movies.loc[similar_movies_list]

    def get_user_recommendations(self, user_id: int, matrix: pd.DataFrame, df_movies: pd.DataFrame) -> pd.DataFrame:
        """
        Raccomanda film a un utente dato utilizzando il filtraggio collaborativo user-based.
        """
        if user_id not in matrix.index:
            raise ValueError(f"User ID {user_id} non trovato nella matrice.")

        all_users = matrix.index

        # 1. Estrae il vettore dei rating dell'utente target
        user_features = matrix.loc[user_id].values.reshape(1, -1)

        # 2. Trova gli utenti simili usando il modello NearestNeighbors
        distances, pos_indexes = self.model_user.kneighbors(user_features)

        # 3. Ricava gli ID degli utenti simili utilizzando gli indici posizionali
        similar_users = [all_users[pos_idx] for pos_idx in pos_indexes.squeeze().tolist()[1:]]
        dist = pd.Series(distances.squeeze().tolist()[1:], index=similar_users)
        print(f"#Utenti simili per user {user_id}: {similar_users} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in dist.items()]}")

        # 4. Recupera i film già visti dall'utente target
        seen_mask = matrix.loc[user_id] > 0.0
        seen_movies = matrix.loc[user_id][seen_mask].index
        print(f"Film visti dall'utente {user_id}: {df_movies.loc[seen_movies] [['title', 'genres']]}")
        print(f"L'utente {user_id} ha visto (numero: {len(seen_movies)}) i seguenti movie_ids: {seen_movies.tolist()}\n")

        # 5. Rimuove i film già visti dall'utente target dalla lista dei raccomandabili
        similars_df = matrix.loc[similar_users]
        similars_df = similars_df.loc[:, ~similars_df.columns.isin(seen_movies)]

        # 6. Calcola la media pesata per colonne e ordina i risultati in ordine decrescente
        mean_weighted_vec = matrix.loc[similar_users].apply(lambda x: np.average(x, weights=1 - dist.to_numpy()))
        mean_movies_df = mean_weighted_vec.to_frame(name="values").sort_values(by="values", ascending=False)
        return mean_movies_df.merge(df_movies, how="left", on="movieId")


def main():
    # Carica il dataset MovieLens
    df_movies, df_ratings, df_tags = load_movielens_data("../dataset/")

    # Crea la matrice utenti-film pivot (userId x movieId)
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Numero di utenti totali: {len(utility_matrix.index)}")
    print(f"# Numero di movies totali: {len(utility_matrix.columns)}")
    print(f"# Utility ratings-matrix: {utility_matrix.shape}")

    NN = 10  # Numero di vicini da considerare
    K = 10  # Numero di raccomandazioni da restituire

    # Inizializza il modello NearestNeighbors con metrica di correlazione di Pearson
    knn_model_pearson_item = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN, n_jobs=-1)
    knn_model_pearson_user = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN, n_jobs=-1)

    # Istanzia il Recommender con il modello KNN
    recomm = CollaborativeRecommender(knn_model_pearson_item, knn_model_pearson_user)
    recomm.fit_user_model(utility_matrix, re_fit=True)
    recomm.fit_item_model(utility_matrix, re_fit=True)

    # Esempio di utilizzo dei metodi di raccomandazione:
    # temp_movie_id = 4306
    # print("Raccomandazioni item-based:")
    # print(recomm.get_item_recommendations(temp_movie_id, utility_matrix, df_movies).head(K), "\n")

    temp_user_id = 406

    print("Raccomandazioni user-based:")
    print(recomm.get_user_recommendations(temp_user_id, utility_matrix, df_movies).head(K), "\n")

    # print("Raccomandazioni ibride:")
    # print(recomm.hybrid_recommender(user_id=temp_user_id, matrix=matrix, df_movies=df_movies, n_recs=10, refit=False), "\n")

    # print("Raccomandazioni ibride2:")
    # print(recomm.hybrid_recommender_mean(user_id=temp_user_id, matrix=matrix, df_movies=df_movies, n_recs=10, refit=False), "\n")


if __name__ == "__main__":
    main()
