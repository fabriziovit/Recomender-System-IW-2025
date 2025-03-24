import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


class CollaborativeRecommender:

    def __init__(
        self,
        model_item: NearestNeighbors,
        model_user: NearestNeighbors,
        user_similarity_matrix: pd.DataFrame = None,
        utility_matrix: pd.DataFrame = None,
    ):
        """Inizializza il Recommender con un modello di machine learning."""
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

        if utility_matrix is not None:
            if not isinstance(utility_matrix, pd.DataFrame):
                raise ValueError(f"utility matrix must be a pd.DataFrame")
            self.utility_matrix = utility_matrix

        self._transposed_matrix = None
        self._is_fitted_on_matrix_T_item = False
        self._is_fitted_on_matrix_user = False
        self.dist_users = None
        self.dist_items = None

    def fit_item_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """Addestra il modello item-based su matrix.T se necessario o se forzato."""
        if not self._is_fitted_on_matrix_T_item or re_fit:
            self._transposed_matrix = matrix.T
            self.model_item.fit(self._transposed_matrix)
            self._is_fitted_on_matrix_T_item = True
            print("# Modello model_item addestrato su matrix.T (movies x users).")
        else:
            print("# Modello model_item già addestrato su matrix.T (movies x users): Salto l'addestramento.")

    def fit_user_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """Addestra il modello user-based su matrix se necessario o se forzato."""
        if not self._is_fitted_on_matrix_user or re_fit:
            self.model_user.fit(matrix)
            self._is_fitted_on_matrix_user = True
            print("# Modello model_user addestrato su matrix (users x movies).")
        else:
            print("# Modello model_user già addestrato su matrix (users x movies): Salto l'addestramento.")

    def get_item_recommendations(self, movie_id: int, df_movies: pd.DataFrame) -> pd.DataFrame:
        """Raccomanda film simili a un film dato utilizzando il filtraggio collaborativo item-based."""

        if movie_id not in self._transposed_matrix.index:
            print(f"Movie ID {movie_id} non trovato nella matrice.")
            raise ValueError(f"Movie ID {movie_id} non trovato nella matrice.")

        all_movie_ids = self._transposed_matrix.index

        # Seleziona le caratteristiche del film specificato dalla matrice trasposta (movieId x userId)
        movie_features = self._transposed_matrix.loc[movie_id].values.reshape(1, -1)

        # Trova i film più simili usando il modello NearestNeighbors
        distances, pos_indexes = self.model_item.kneighbors(movie_features)

        # Ricava gli ID dei film simili utilizzando gli indici posizionali
        similar_movies_list = [all_movie_ids[pos_idx] for pos_idx in pos_indexes.squeeze().tolist()[1:]]

        # Salva le distanze item-item per il film specificato
        self.dist_items = pd.Series(distances.squeeze().tolist()[1:], index=similar_movies_list)
        print(f"\nFilm simili trovati per il film: {df_movies.loc[movie_id, 'title']} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in self.dist_items.items()]}")

        # Crea un DataFrame con le raccomandazioni dei film
        return df_movies.loc[similar_movies_list]

    def get_user_recommendations(self, user_id: int, matrix: pd.DataFrame, df_movies: pd.DataFrame) -> pd.DataFrame:
        """Raccomanda film a un utente dato utilizzando il filtraggio collaborativo user-based."""
        if user_id not in matrix.index:
            raise ValueError(f"User ID {user_id} non trovato nella matrice.")

        all_users = matrix.index

        # Estrae il vettore dei rating dell'utente target
        user_features = matrix.loc[user_id].values.reshape(1, -1)

        # Trova gli utenti simili usando il modello NearestNeighbors
        distances, pos_indexes = self.model_user.kneighbors(user_features)

        # Ricava gli ID degli utenti simili utilizzando gli indici posizionali
        similar_users = [all_users[pos_idx] for pos_idx in pos_indexes.squeeze().tolist()[1:]]

        # Salava le distanze user-user per l'utente specificato
        self.dist_users = pd.Series(distances.squeeze().tolist()[1:], index=similar_users)

        print(f"\nUtenti simili per user {user_id}: {similar_users} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in self.dist_users.items()]}")

        # Recupera i film già visti dall'utente target
        seen_mask = matrix.loc[user_id] > 0.0
        seen_movies = matrix.loc[user_id][seen_mask].index

        # Recupero i film visti dagli utenti simili e Rimuove i film già visti dall'utente target
        similars_movies_df = matrix.loc[similar_users]
        similars_movies_df = similars_movies_df.loc[:, ~similars_movies_df.columns.isin(seen_movies)]

        # Vettore contenente la media pesata dei rating dei film raccomandati e Ordina i film raccomandati in base alla media pesata dei rating
        mean_weighted_vec = similars_movies_df.apply(lambda x: np.average(x, weights=1 - self.dist_users.to_numpy()))
        mean_movies_df = mean_weighted_vec.to_frame(name="values").sort_values(by="values", ascending=False)

        return mean_movies_df.merge(df_movies, how="left", on="movieId")

    # ************************************************************************************************ #

    def _get_k_similar_users(self, user_id: int, NN: int):
        """Ottieni utenti simili usando la matrice di similarità."""
        if self.user_similarity_matrix is None:
            raise ValueError("User similarity matrix not computed or provided.")
        if user_id not in self.user_similarity_matrix.index:
            raise ValueError(f"User ID {user_id} not found in the user similarity matrix.")

        user_similarities = self.user_similarity_matrix.loc[user_id]
        similar_users = user_similarities.drop(user_id).sort_values(ascending=False)  # Escludi se stesso e ordina

        similar_users = similar_users.head(NN)  # Prendi solo i primi NN utenti simili
        return list(similar_users.items())  # Restituisci lista di (user_id, similarity) per l'utenete target

    def _predict_rating(self, user_id: int, movie_id: int, NN: int, matrix: pd.DataFrame) -> float:
        """Predice il rating per un utente e un film usando una matrice di similarità utente-utente in maniera mean-centered."""
        if self.user_similarity_matrix is None:
            raise ValueError("User similarity matrix not provided.")
        if user_id not in matrix.index or movie_id not in matrix.columns:
            raise ValueError("User ID or Movie ID not found in the train matrix.")

        # Ottieni gli utenti simili e le loro similarità
        similar_users_tlp: tuple = self._get_k_similar_users(user_id, NN=NN)
        if not similar_users_tlp:
            raise ValueError("No similar users found.")
        similar_user_ids = [uid for uid, sim in similar_users_tlp]
        similarities = np.array([sim for uid, sim in similar_users_tlp])

        # Calcola la media dell'utente target (ignorando gli 0)
        target_user_ratings = matrix.loc[user_id]
        target_user_valid_ratings = target_user_ratings[target_user_ratings > 0.0]
        target_user_mean = target_user_valid_ratings.mean()

        # Ottieni i rating degli utenti simili per il film target
        similar_users_ratings = matrix.loc[similar_user_ids, movie_id].values
        # Calcola le medie degli utenti simili (ignorando gli 0) per ciascun utente simile
        similar_users_means = matrix.loc[similar_user_ids].replace(0, np.nan).mean(axis=1).values

        # Crea una maschera per i rating validi
        valid_mask = similar_users_ratings > 0.0

        if np.any(valid_mask):
            valid_similarities = similarities[valid_mask]
            valid_ratings = similar_users_ratings[valid_mask]
            valid_means = similar_users_means[valid_mask]

            # Calcola le deviazioni mean-centered per gli utenti simili
            weighted_deviation = np.sum(valid_similarities * (valid_ratings - valid_means))
            similarity_sum = np.sum(np.abs(valid_similarities))

            prediction_centered = weighted_deviation / similarity_sum
            return target_user_mean + prediction_centered
        else:
            # Fallback: se nessun rating valido, usa la media dell'utente target
            if not target_user_valid_ratings.empty:
                return target_user_mean
            raise ValueError("No valid ratings found.")

    def _get_predictions_on_train(self, user_id: int, train_matrix: pd.DataFrame, NN, exclude: bool = False) -> pd.Series:
        """Calcola le predizioni per un utente specifico."""
        if user_id not in train_matrix.index:
            raise ValueError(f"User ID {user_id} not found in the matrix.")
        predicted_ratings_list = []
        print(f"Predizione per utente {user_id}...")
        seen_movies_ids = train_matrix.loc[user_id][train_matrix.loc[user_id] > 0.0].index
        for movie_id in train_matrix.columns:
            if exclude and movie_id in seen_movies_ids:
                continue  # Salta i film già visti
            predicted_rating = self._predict_rating(user_id, movie_id, NN, train_matrix)
            predicted_ratings_list.append({"movieId": movie_id, "predicted_rating": predicted_rating})
        predictions_df = pd.DataFrame(predicted_ratings_list).set_index("movieId").rename(columns={"predicted_rating": "values"})
        return predictions_df.sort_values("values", ascending=False)

    def compute_predictions_on_train(self, NN, train_matrix) -> dict:
        """Calcola le predizioni sul training set"""
        all_user_predictions_evaluation = {}
        for user_id in train_matrix.index:
            all_user_predictions_evaluation[user_id] = self._get_predictions_on_train(user_id, train_matrix, NN, exclude=False)
        return all_user_predictions_evaluation

    def get_prediction(self, user_id: int, movie_id: int, NN: int = 20) -> pd.Series:
        """Calcola le predizioni per un utente specifico."""
        predicted_rating = self._predict_rating(user_id, movie_id, NN=NN, matrix=self.utility_matrix)
        return predicted_rating

    # ************************************************************************************************ #
    def __get_similar_users(self, user_id: int, matrix: pd.DataFrame) -> list:
        """Ottieni utenti simili usando la matrice di similarità."""
        all_user_ids = matrix.index
        user_feature = matrix.loc[user_id].values.reshape(1, -1)
        distances, pos_indexes = self.model_user.kneighbors(user_feature)

        similar_users = all_user_ids[pos_indexes.squeeze()[1:]]

        # Salva le NN distanze user-user per l'utente specificato
        self.dist_users = pd.Series(distances.squeeze()[1:], index=similar_users)
        return similar_users

    def _knn_predict_rating(self, user_id: int, movie_id: int, matrix: pd.DataFrame, similar_users=None, similar_mean=None) -> float:
        """Predice il rating per un utente e un film utilizando knn (Vettorizzata)."""
        if user_id not in matrix.index:
            raise ValueError(f"User ID {user_id} not found in the matrix.")

        # Calcolo della mean dell'utenza target
        target_user_ratings = matrix.loc[user_id]
        target_user_mean = target_user_ratings[target_user_ratings > 0.0].mean()

        # Recupero gli utenti simili solo se non sono già stati passati
        if similar_users is None:
            similar_users = self.__get_similar_users(user_id, matrix)
            similar_users_df = matrix.loc[similar_users]
            valid_ratings_mask = similar_users_df > 0.0
            similar_mean = similar_users_df.where(valid_ratings_mask).mean(axis=1)
        else:
            similar_users_df = matrix.loc[similar_users]

        # Recupero i rating degli utenti simili per il film specificato movie_id
        similar_users_for_movie = similar_users_df[movie_id]
        valid_mask_for_movie = similar_users_for_movie > 0.0

        # Se c'è almeno un rating valido > 0.0
        if np.any(valid_mask_for_movie):
            valid_ratings = similar_users_for_movie[valid_mask_for_movie]
            valid_similarities = 1 - self.dist_users[valid_mask_for_movie]

            weighted_deviation = np.sum(valid_similarities * (valid_ratings - similar_mean[valid_mask_for_movie]))
            prediction_centered = weighted_deviation / np.abs(np.sum(valid_similarities))
            return target_user_mean + prediction_centered
        else:
            # Usa i calcoli già fatti
            if np.any(target_user_ratings > 0.0):
                return target_user_mean
            raise ValueError(f"No valid ratings found for User ID {user_id}.")

    def knn_get_prediction(self, user_id: int, movie_id: int) -> float:
        """Calcola le predizioni per un utente specifico."""
        return self._knn_predict_rating(user_id, movie_id, self.utility_matrix)

    def _knn_get_predictions_on_train(self, user_id: int, train_matrix: pd.DataFrame, NN: int, exclude: bool = False) -> pd.Series:
        """Calcola le predizioni per un utente specifico."""
        if user_id not in train_matrix.index:
            raise ValueError(f"User ID {user_id} not found in the matrix.")

        print(f"Predizione per utente {user_id}...")

        # Pre-calcola gli utenti simili una sola volta
        similar_users = self.__get_similar_users(user_id, train_matrix)
        similar_users_df = train_matrix.loc[similar_users]
        valid_ratings_mask = similar_users_df > 0.0
        similar_mean = similar_users_df.where(valid_ratings_mask).mean(axis=1)

        # Pre-filtra i film da prevedere
        user_ratings = train_matrix.loc[user_id]
        seen_movies_ids = set(user_ratings[user_ratings > 0.0].index) if exclude else set()
        movies_to_predict = [movie_id for movie_id in train_matrix.columns if movie_id not in seen_movies_ids]

        # Calcola le predizioni passando gli utenti simili e le loro medie
        predicted_ratings_list = [
            {"movieId": movie_id, "predicted_rating": self._knn_predict_rating(user_id, movie_id, train_matrix, similar_users, similar_mean)} for movie_id in movies_to_predict
        ]

        predictions_df = pd.DataFrame(predicted_ratings_list).set_index("movieId").rename(columns={"predicted_rating": "values"})
        return predictions_df.sort_values("values", ascending=False)

    def knn_compute_predictions_on_train(self, NN, train_matrix) -> dict:
        """Calcola le predizioni sul training set"""
        return {user_id: self._knn_get_predictions_on_train(user_id, train_matrix, NN, exclude=False) for user_id in train_matrix.index}

    # ************************************************************************************************ #


# if __name__ == "__main__":
#     # Test CollaborativeRecommender Prediction
#     from utils import load_movielens_data, pearson_similarity, pearson_distance

#     # Carica il dataset MovieLens
#     df_movies, df_ratings, df_tags = load_movielens_data("dataset/")

#     all_movies_id = df_movies.index

#     # Crea la utility matrix
#     utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
#     print(f"Numero di utenti totali: {len(utility_matrix.index)}")
#     print(f"Numero di movies totali: {len(utility_matrix.columns)}")
#     print(f"Utility ratings-matrix.shape: {utility_matrix.shape}")

#     NN: int = 20
#     # Inizializza il modello NearestNeighbors per il filtraggio collaborativo item-based
#     model_item = NearestNeighbors(n_neighbors=NN + 1, algorithm="brute", metric=pearson_distance)
#     model_user = NearestNeighbors(n_neighbors=NN + 1, algorithm="brute", metric=pearson_distance)

#     # Inizializza il Recommender
#     recommender = CollaborativeRecommender(model_item, model_user, utility_matrix=utility_matrix)
#     # Addestra il modello user-based
#     recommender.fit_user_model(utility_matrix)

#     # Calcola le predizioni per un utente specifico
#     temp_user_id = 1
#     temp_movie_id = 2
#     prediction = recommender.knn_get_prediction(temp_user_id, temp_movie_id)
#     print(f"Predizione per user {temp_user_id} e movie {temp_movie_id}: {prediction}")
