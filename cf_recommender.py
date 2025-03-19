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
        self._is_fitted_on_matrix_user = False

        self.dist_users = None

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
        dist_items = pd.Series(distances.squeeze().tolist()[1:], index=similar_movies_list)
        print(f"\nFilm simili trovati per il film: {df_movies.loc[movie_id, 'title']} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in dist_items.items()]}")

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

    #! Da commenate
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

    # ******************************************************************************************************** #
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

    def _predict_rating(self, user_id: int, movie_id: int, NN: int) -> float:
        """Predice il rating per un utente e un film usando il training set e similarità utente-utente pre-calcolata (Vettorizzata)."""
        if self.user_similarity_matrix is None or self.train_matrix is None:
            raise ValueError("User similarity matrix or train matrix not provided.")
        if user_id not in self.train_matrix.index or movie_id not in self.train_matrix.columns:
            raise ValueError("User ID or Movie ID not found in the train matrix.")

        similar_users_tlp: tuple = self._get_k_similar_users(user_id, NN=NN)
        if not similar_users_tlp:
            raise ValueError("No similar users found.")  # Se non ci sono utenti simili, non posso fare predizioni

        similar_user_ids = [uid for uid, sim in similar_users_tlp]  # Estrai solo gli user_id simili
        similarities = np.array([sim for uid, sim in similar_users_tlp])  # Estrai le similarità come array NumPy

        # Accedi ai rating gli utenti simili per il film specificato
        similar_users_ratings = self.train_matrix.loc[similar_user_ids, movie_id].values  # Ottieni come array NumPy

        # Crea una maschera booleana per i rating validi (> 0.0)
        valid_mask = similar_users_ratings > 0.0

        if np.any(valid_mask):  # Verifica se ci sono rating validi
            valid_similarities = similarities[valid_mask]  # Filtra similarità corrispondenti ai rating validi
            valid_ratings = similar_users_ratings[valid_mask]  # Filtra rating validi

            weighted_sum = np.sum(valid_similarities * valid_ratings)  # Calcola la somma pesata in modo vettorizzato
            similarity_sum = np.sum(np.abs(valid_similarities))  # Calcola la somma delle similarità in modo vettorizzato
            return weighted_sum / similarity_sum
        else:
            # Se non ci sono valutazioni valide, restituisco errore
            user_ratings = self.train_matrix.loc[user_id]
            valid_ratings = user_ratings[user_ratings > 0.0]
            if not valid_ratings.empty:
                return valid_ratings.mean()
            raise (ValueError("No valid ratings found."))  # Se non ci sono valutazioni valide, restituisco errore

    def _get_predictions_on_train(self, user_id: int, matrix: pd.DataFrame, NN, exclude: bool = False) -> pd.Series:
        """Calcola le predizioni per un utente specifico."""
        if user_id not in matrix.index:
            raise ValueError(f"User ID {user_id} not found in the matrix.")
        predicted_ratings_list = []
        print(f"Predizione per utente {user_id}...")
        seen_movies_ids = self.train_matrix.loc[user_id][self.train_matrix.loc[user_id] > 0.0].index
        for movie_id in matrix.columns:
            if exclude and movie_id in seen_movies_ids:
                continue  # Salta i film già visti
            predicted_rating = self._predict_rating(user_id, movie_id, NN=NN)
            predicted_ratings_list.append({"movieId": movie_id, "predicted_rating": predicted_rating})
        predictions_df = pd.DataFrame(predicted_ratings_list).set_index("movieId").rename(columns={"predicted_rating": "values"})
        return predictions_df.sort_values("values", ascending=False)

    def compute_predictions_on_train(self, NN) -> dict:
        """Calcola le predizioni sul training set"""
        all_user_predictions_evaluation = {}
        for user_id in self.train_matrix.index:
            all_user_predictions_evaluation[user_id] = self._get_predictions_on_train(user_id, self.train_matrix, NN, exclude=False)
        return all_user_predictions_evaluation

    def get_prediction(self, user_id: int, movie_id: int, NN: int = 20) -> pd.Series:
        """Calcola le predizioni per un utente specifico."""
        predicted_rating = self._predict_rating(user_id, movie_id, NN=NN)
        return predicted_rating
