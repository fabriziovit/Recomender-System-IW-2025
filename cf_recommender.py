import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import load_movielens_data, pearson_distance


class CollaborativeRecommender:

    def __init__(
        self,
        model_item: NearestNeighbors,
        model_user: NearestNeighbors,
        user_similarity_matrix: pd.DataFrame = None,
        train_matrix: pd.DataFrame = None,
    ):
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

        if user_similarity_matrix is not None:
            if not isinstance(user_similarity_matrix, pd.DataFrame):
                raise ValueError(f"user_similarity_matrix must be a pd.DataFrame")
            self.user_similarity_matrix = user_similarity_matrix

        if train_matrix is not None:
            if not isinstance(train_matrix, pd.DataFrame):
                raise ValueError(f"train_matrix must be a pd.DataFrame")
            self.train_matrix = train_matrix

        self._transposed_matrix = None

        self._is_fitted_on_matrix_T_item = False
        self.sim_items: pd.Series = None
        self.dist_items: pd.Series = None

        self._is_fitted_on_matrix_user = False
        self.sim_users: pd.Series = None
        self.dist_users: pd.Series = None

    def fit_item_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """
        Addestra il modello item-based su matrix.T se necessario o se forzato.
        """
        if not self._is_fitted_on_matrix_T_item or re_fit:
            print(f"self._is_fitted_on_matrix_T_item: {self._is_fitted_on_matrix_T_item}")
            print(f"re_fit: {re_fit}")
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

    def get_item_recommendations(self, movie_id: int, df_movies: pd.DataFrame) -> pd.DataFrame:
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

        # Salva le distanze item-item per il film specificato
        self.dist_items = pd.Series(distances.squeeze().tolist()[1:], index=similar_movies_list)
        # Salva le similarità item-item per il film specificato
        self.sim_items = pd.Series(1 - self.dist_items, index=similar_movies_list)

        print(f"\nFilm simili trovati per il film: {df_movies.loc[movie_id, 'title']} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in self.dist_items.items()]}")

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

        # Salava le distanze user-user per l'utente specificato
        self.dist_users = pd.Series(distances.squeeze().tolist()[1:], index=similar_users)
        # Salava le similarità user-user per l'utente specificato
        self.sim_users = pd.Series(1 - self.dist_users, index=similar_users)

        print(f"\nUtenti simili per user {user_id}: {similar_users} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in self.dist_users.items()]}")

        # 4. Recupera i film già visti dall'utente target
        seen_mask = matrix.loc[user_id] > 0.0
        seen_movies = matrix.loc[user_id][seen_mask].index

        # 5. Recupero i film visti dagli utenti simili e Rimuove i film già visti dall'utente target
        similars_movies_df = matrix.loc[similar_users]
        similars_movies_df = similars_movies_df.loc[:, ~similars_movies_df.columns.isin(seen_movies)]

        # 6. Vettore contenente la media pesata dei rating dei film raccomandati e Ordina i film raccomandati in base alla media pesata dei rating
        mean_weighted_vec = similars_movies_df.apply(lambda x: np.average(x, weights=1 - self.dist_users.to_numpy()))
        mean_movies_df = mean_weighted_vec.to_frame(name="values").sort_values(by="values", ascending=False)

        return mean_movies_df.merge(df_movies, how="left", on="movieId")

    def get_mean_centered_predictions(self, df_ratings: pd.DataFrame, sim_scores: pd.Series, user_id: int, curr_movie_id: int) -> float:

        # Converti sim_scores in DataFrame per operazioni vettorializzate
        df_sim_scores = sim_scores.rename("similarity").reset_index()
        df_sim_scores.rename(columns={"index": "userId"}, inplace=True)

        # 1. Calcolo la media delle valutazioni per ogni utente
        users_means: pd.Series = df_ratings.groupby("userId")["rating"].mean().rename("user_mean")
        df_ratings_with_mean: pd.DataFrame = df_ratings.merge(users_means, on="userId")

        # 2. Calcolo il numero di valutazioni per ogni utente
        user_ratings_count = df_ratings.groupby("userId")["rating"].count().rename("num_ratings")

        # 2. Seleziono le valutazioni degli utenti simili per il film selezionato
        ratings_similar_df = df_ratings_with_mean[df_ratings_with_mean["movieId"] == curr_movie_id].merge(df_sim_scores, on="userId", how="inner")

        # 3. Aggiungo il numero di valutazioni per ogni utente
        ratings_similar_df = ratings_similar_df.merge(user_ratings_count, on="userId")

        # 4. Aggiungo la similarità normalizzata
        ratings_similar_df["adjusted_similarity"] = ratings_similar_df["similarity"] / (ratings_similar_df["num_ratings"] + 1)

        if not ratings_similar_df.empty:
            # Centro le valutazioni sottraendo la media dell'utente
            ratings_similar_df["mean_centered_rating"] = ratings_similar_df["rating"] - ratings_similar_df["user_mean"]

            # Calcolo la media pesata delle valutazioni centrate
            weighted_sum = np.dot(ratings_similar_df["mean_centered_rating"], ratings_similar_df["adjusted_similarity"])
            prediction_mean_centered = weighted_sum / ratings_similar_df["adjusted_similarity"].sum()

            # Riporto la predizione alla scala originale sommando la media dell'utente target
            user_target_mean = users_means.get(user_id, df_ratings["rating"].mean())  # Se l'utente target non ha media, uso la media globale
            final_prediction = prediction_mean_centered + user_target_mean
            return final_prediction
        else:
            raise ValueError("Nessun utente simile ha valutato il film selezionato")


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
