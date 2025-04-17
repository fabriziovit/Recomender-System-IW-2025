import time
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CollaborativeRecommender:

    def __init__(self, model_item: NearestNeighbors, model_user: NearestNeighbors):
        """Inizializza il Recommender con un modello di machine learning."""
        if not isinstance(model_item, NearestNeighbors):
            raise ValueError(f"model_item must be NearestNeighbors")
        self.model_item = model_item

        if not isinstance(model_user, NearestNeighbors):
            raise ValueError(f"model_user must be NearestNeighbors")
        if model_user.metric != "cosine":
            raise ValueError("model_user metric must be 'cosine' for mean-centered approach.")
        self.model_user = model_user

        self.utility_matrix = None
        self.centered_utility_matrix = None
        self.user_means = None  # Memorizza le medie degli utenti

        self._transposed_matrix = None
        self._is_fitted_on_matrix_T_item = False
        self._is_fitted_on_matrix_user = False
        self.dist_users = None
        self.dist_items = None

    def _compute_mean_centered_matrix_user_based(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Calcola la matrice centrata sulla media, ignorando 0.0 per il calcolo della media."""
        # Calcola la media per ogni utente non considerando gli 0.0 (sostituendo 0.0 con NaN)
        self.user_means = matrix.replace(0.0, np.nan).mean(axis=1)
        # Sostituisci i valori NaN con 0.0 (assenza di rating)
        self.user_means.fillna(0.0, inplace=True)

        # Sottrai la media utente da ogni rating valido.
        # .sub() sottrae la Series user_means riga per riga.
        # .where() applica la sottrazione solo dove valid_mask è True, altrimenti mette 0.
        valid_mask = matrix > 0.0
        cenetered_matrix = matrix.sub(self.user_means, axis=0).where(valid_mask, 0)
        logging.info("# Matrice mean-centered calcolata.")
        return cenetered_matrix

    def fit_item_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """Addestra il modello item-based su matrix.T se necessario o se forzato."""
        if not self._is_fitted_on_matrix_T_item or re_fit:
            self._transposed_matrix = matrix.T
            self.model_item.fit(self._transposed_matrix)
            self._is_fitted_on_matrix_T_item = True
            logging.info("# Modello model_item addestrato su matrix.T (movies x users).")
        else:
            logging.info("# Modello model_item già addestrato su matrix.T (movies x users): Salto l'addestramento.")

    def fit_user_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """Addestra il modello user-based su matrix se necessario o se forzato."""

        # NOTA: Si effettua il fit del modello sulla matrice mean-centered utilizzando la metrica di coseno
        # per calcolare le distanze tra gli utenti che è equivalente a calcolare la Correlazione di Pearson
        # quando si calcolano le predizioni utilizzando il metodo _predict_rating.

        if not self._is_fitted_on_matrix_user or re_fit:
            # 1. Memorizza la matrice utility_matrix
            self.utility_matrix = matrix
            # 2. Calcola e Memorizza la matrice centrata sulla media
            self.centered_utility_matrix = self._compute_mean_centered_matrix_user_based(self.utility_matrix)
            logging.info("# Matrice utility_matrix e matrice mean-centered calcolate.")
            # 3. Addestra il modello user-based sulla matrice centrata
            self.model_user.fit(self.centered_utility_matrix)
            self._is_fitted_on_matrix_user = True
            logging.info("# Modello model_user addestrato su matrix_centered (users x movies).")
        else:
            logging.info("# Modello model_user già addestrato su matrix_centered (users x movies): Salto l'addestramento.")

    # ************************************************************************************************ #

    def get_item_recommendations(self, movie_id: int, df_movies: pd.DataFrame) -> pd.DataFrame:
        """Raccomanda film simili a un film dato utilizzando il filtraggio collaborativo item-based."""

        if not self._is_fitted_on_matrix_T_item or self._transposed_matrix is None:
            raise RuntimeError("Modello item non addestrato o matrice non disponibile.")
        if movie_id not in self._transposed_matrix.index:
            logging.info(f"Movie ID {movie_id} non trovato nella matrice.")
            raise ValueError(f"Movie ID {movie_id} non trovato nella matrice.")

        all_movie_ids = self._transposed_matrix.index

        # Seleziona le caratteristiche del film specificato dalla matrice trasposta (movieId x userId)
        movie_features = self._transposed_matrix.loc[movie_id].values.reshape(1, -1)

        # Trova i film più simili usando il modello NearestNeighbors
        distances, pos_indexes = self.model_item.kneighbors(movie_features)

        logging.info(f"Distanze per il film {movie_id}: {distances.squeeze().tolist()[1:]}")
        logging.info(f"Indici posizionali per il film {movie_id}: {pos_indexes.squeeze().tolist()[1:]}")

        # Ricava gli ID dei film simili utilizzando gli indici posizionali
        similar_movies_list = [all_movie_ids[pos_idx] for pos_idx in pos_indexes.squeeze().tolist()[1:]]
        logging.info(f"Similar movies found for movie {movie_id}: {similar_movies_list}")

        # Salva le distanze item-item per il film specificato
        self.dist_items = pd.Series(distances.squeeze().tolist()[1:], index=similar_movies_list)
        logging.info(
            f"\nFilm simili trovati per il film: {df_movies.loc[movie_id, 'title']} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in self.dist_items.items()]}"
        )

        # Crea un DataFrame con le raccomandazioni dei film
        return df_movies.loc[similar_movies_list]

    # ************************************************************************************************ #

    def _get_recommendation_values(self, user_id: int) -> pd.Series:
        """Calcola le raccomandazioni per un utente specifico."""

        # 1. Pre-calcola gli utenti simili e le loro medie una sola volta
        similar_users_ids = self._get_similar_users(user_id)  # Usa modello/matrice centrata

        # Pre-filtra i film da prevedere
        user_ratings = self.utility_matrix.loc[user_id]

        # Escludi i film già visti dall'utente
        seen_movies_ids = set(user_ratings[user_ratings > 0.0].index)
        movies_to_predict = [movie_id for movie_id in self.utility_matrix.columns if movie_id not in seen_movies_ids]

        # 3. Calcola le predizioni passando gli utenti simili e le loro medie precalcolate
        predicted_ratings_list = []
        for movie_id in movies_to_predict:
            pred = self._predict_rating(user_id, movie_id, similar_users_ids)
            predicted_ratings_list.append({"movieId": movie_id, "predicted_rating": pred})

        # 4. Crea e ordina il DataFrame delle predizioni
        predictions_df = pd.DataFrame(predicted_ratings_list).set_index("movieId").rename(columns={"predicted_rating": "values"})
        return predictions_df.sort_values("values", ascending=False)

    def get_user_recommendations(self, user_id: int, df_movies: pd.DataFrame = None) -> pd.DataFrame:
        """Restituisce le raccomandazioni per un utente specifico."""

        if not self._is_fitted_on_matrix_user or self.utility_matrix is None or self.centered_utility_matrix is None:
            raise RuntimeError("Modello user non addestrato o matrice non disponibile.")
        if user_id not in self.utility_matrix.index:
            raise ValueError(f"User ID {user_id} non trovato nella matrice.")

        # Calcola le predizioni per l'utente specificato
        predictions_df = self._get_recommendation_values(user_id)

        if df_movies is None:
            # Se non è fornito un DataFrame di film, restituisci solo le predizioni
            return predictions_df

        # Unisce le raccomandazioni con il DataFrame dei film su movieId
        # logging.info(f"Predizioni per l'utente {user_id}:\n{predictions_df.merge(df_movies, how='left', on='movieId')}")
        return predictions_df.merge(df_movies, how="left", on="movieId")

    # ************************************************************************************************ #
    def _get_similar_users(self, user_id: int) -> list:
        """Ottieni utenti simili usando il modello KNN addestrato sulla matrice mean-centered."""

        # Assicurati che i modelli siano addestrati e le matrici disponibili
        if not self._is_fitted_on_matrix_user or self.utility_matrix is None or self.centered_utility_matrix is None:
            raise RuntimeError("Modello user non addestrato o matrici non disponibili.")

        # 1. Ottieni il vettore delle feature CENTRATE per l'utente target
        user_centered_feature = self.centered_utility_matrix.loc[user_id].values.reshape(1, -1)

        # 2. Trova vicini basati sulla distanza coseno nel spazio centrato
        distances, pos_indexes = self.model_user.kneighbors(user_centered_feature)

        # Gli indici si riferiscono all'ordine della matrice (originale o centrata, sono uguali)
        all_user_ids = self.utility_matrix.index
        similar_users_ids = all_user_ids[pos_indexes.squeeze()[1:]]

        # Salva le distanze COSENO user-user per l'utente specificato
        # La distanza coseno è tra 0 (identici) e 2 (opposti), 1 (ortogonali)
        self.dist_users = pd.Series(distances.squeeze()[1:], index=similar_users_ids)

        return similar_users_ids

    def _predict_rating(self, user_id: int, movie_id: int, similar_users_ids=None) -> float:
        """
        Predice il rating per un utente e un film utilizzando knn e mean centering.
        Usa la MATRICE ORIGINALE per i rating e le medie, ma i vicini e le loro
        distanze (coseno) sono stati trovati usando la matrice centrata.
        """

        if user_id not in self.utility_matrix.index:
            raise ValueError(f"User ID {user_id} not found in the original matrix.")
        if movie_id not in self.utility_matrix.columns:
            raise ValueError(f"Movie ID {movie_id} not found in the original matrix.")

        # 1. Recupero della mean dell'utente target
        target_user_mean = self.user_means.loc[user_id]
        if target_user_mean is None:
            raise ValueError(f"Mean rating for user ID {user_id} not found.")

        # 2. Calcolo gli utenti simili se non disponibili
        if similar_users_ids is None:
            similar_users_ids = self._get_similar_users(user_id)

        if len(similar_users_ids) <= 0:
            logging.info(f"Nessun utente simile trovato per l'utente {user_id}.")
            return target_user_mean

        else:
            # 3. Recupera le medie degli utenti simili calcolate in fit_user_model
            similar_mean_values = self.user_means.loc[similar_users_ids]

            # 4. Recupero le valutazioni degli utenti simili
            similar_users_df = self.utility_matrix.loc[similar_users_ids]

            # 5. Recupero le valutazioni degli utenti simili per il film specificato (movie_id)
            similar_users_for_movie = similar_users_df[movie_id]
            valid_mask_for_movie = similar_users_for_movie > 0.0

            # Se c'è almeno un rating valido > 0.0 tra i vicini per questo film
            if valid_mask_for_movie.any():
                # Seleziona i rating validi dei vicini per il film
                valid_ratings = similar_users_for_movie[valid_mask_for_movie]

                # Seleziona le distanze coseno corrispondenti ai vicini validi
                valid_distances = self.dist_users[valid_mask_for_movie]

                # Calcola le similarità coseno (1 - distanza coseno)
                valid_similarities = 1.0 - valid_distances

                # Seleziona le medie dei vicini validi
                valid_similar_means = similar_mean_values[valid_mask_for_movie]

                # Calcola la somma pesata: similarità * (rating_vicino - media_vicino)
                weighted_sum = np.sum(valid_similarities * (valid_ratings - valid_similar_means))

                # Somma dei valori assoluti delle similarità come denominatore
                sum_of_similarities = np.sum(np.abs(valid_similarities))

                # Evita divisione per zero
                if sum_of_similarities == 0:
                    prediction_centered = 0  # Nessuna deviazione se le similarità sono tutte zero
                else:
                    prediction_centered = weighted_sum / sum_of_similarities

                # Predizione finale = media_utente_target + prediction_centered
                return target_user_mean + prediction_centered

            else:
                # Se non ci sono rating validi per questo film, restituisci la media dell'utente target
                # logging.info(f"Nessun rating valido trovato per il film {movie_id} da parte degli utenti simili.")
                return target_user_mean

    def get_prediction_value_clipped(self, user_id: int, movie_id: int, min_rating: float = 0.5, max_rating: float = 5.0) -> float:
        """Calcola la predizione per un utente e un film specifici. (clipped between 0.5 and 5.0)"""
        predicted_rating = self._predict_rating(user_id, movie_id, None)
        return np.clip(predicted_rating, min_rating, max_rating)

    # ************************************************************************************************ #
    def evaluate_mae_rmse(self, test_matrix: pd.DataFrame, min_rating: float = 0.5, max_rating: float = 5.0):
        """Valuta il modello sul test set calcolando MAE e RMSE."""

        # Assicurati che il modello user sia stato addestrato
        if not self._is_fitted_on_matrix_user or self.utility_matrix is None:
            raise RuntimeError("Modello user non addestrato. Chiamare prima fit_user_model sul training set.")

        true_ratings = []
        predicted_ratings = []
        skipped_count = 0
        start_time = time.time()

        # *** --- Iterazione Efficiente sul Test Set --- ***#
        # Usiamo stack() per ottenere una Series con (userId, movieId) come indice e il rating come valore,
        # ignorando automaticamente gli zeri.
        test_ratings_series = test_matrix[test_matrix > 0].stack()
        total_ratings_to_predict = len(test_ratings_series)

        logging.info(f"# Inizio valutazione su {total_ratings_to_predict} rating nel test set...")

        # Iteriamo sulla Series risultante
        for idx, true_rating in test_ratings_series.items():
            logging.info(f"Predizione per {idx}...")
            user_id, movie_id = idx[0], idx[1]  # Estrai userId e movieId dall'indice

            # Controlla se user/movie esistono nella matrice di training
            if user_id not in self.utility_matrix.index or movie_id not in self.utility_matrix.columns:
                skipped_count += 1
                continue  # Salta se non presenti nel training

            # Calcola la predizione per la specifica coppia (user, movie)
            try:
                predicted_rating_raw = self._predict_rating(user_id, movie_id)
                # Applica il clipping qui, prima di calcolare le metriche
                predicted_rating_clipped = np.clip(predicted_rating_raw, min_rating, max_rating)

                # Aggiungi i valori alle liste
                true_ratings.append(true_rating)
                predicted_ratings.append(predicted_rating_clipped)

            except ValueError as e:
                skipped_count += 1
                continue
            except Exception as e:  # Cattura altre eccezioni impreviste
                logging.info(f"Eccezione inattesa per ({user_id}, {movie_id}): {e}")
                skipped_count += 1
                continue

        end_time = time.time()

        logging.info(f"# Valutazione completata in {end_time - start_time:.2f} secondi.")
        if skipped_count > 0:
            logging.info(f"#   Skipped {skipped_count} rating (user/movie non nel training o errore).")
        if not true_ratings:  # Se non ci sono rating validi su cui valutare
            logging.info("#   Nessun rating valido trovato per la valutazione.")
            return np.nan, np.nan

        # Calcola MAE e RMSE usando le liste raccolte
        mae = mean_absolute_error(true_ratings, predicted_ratings)  # MAE
        rmse = root_mean_squared_error(true_ratings, predicted_ratings)  # RMSE

        return mae, rmse

    def evaluate_precision_recall(self, test_matrix: pd.DataFrame, K_list: list[int], relevant_threshold: float) -> dict:
        """
        Valuta il modello sul test set calcolando Precision@K e Recall@K.
        """
        # Assicurati che il modello user sia stato addestrato
        if not self._is_fitted_on_matrix_user or self.utility_matrix is None:
            raise RuntimeError("Modello user non addestrato. Chiamare prima fit_user_model sul training set.")

        # Dizionari per accumulare le somme di precision e recall per ogni K
        precision_sum = defaultdict(float)
        recall_sum = defaultdict(float)
        evaluated_user_count = defaultdict(int)  # Contatore utenti validi per ogni K
        max_K = max(K_list)  # Il K massimo necessario per le predizioni

        processed_users = 0
        start_time = time.time()

        # Iteriamo sugli utenti presenti nel test set
        test_users = test_matrix.index
        # Opzionale: considera solo utenti presenti anche nel training
        valid_test_users = test_users.intersection(self.utility_matrix.index)
        total_users_to_evaluate = len(valid_test_users)

        logging.info(f"# Inizio valutazione Precision/Recall per K={K_list}, soglia={relevant_threshold}...")
        logging.info(f"# Valutazione su {total_users_to_evaluate} utenti presenti sia in test che in train.")

        for user_id in valid_test_users:
            logging.info(f"Valutazione di Precision e Recall per l'utente {user_id}...")
            processed_users += 1

            # 1. Trova gli item rilevanti per l'utente nel test set
            test_user_ratings = test_matrix.loc[user_id]
            relevant_items_in_test = set(test_user_ratings[test_user_ratings >= relevant_threshold].index)

            # Se l'utente non ha item rilevanti nel test set, non possiamo calcolare Recall
            # e la Precision non è molto significativa. Saltiamo l'utente.
            if not relevant_items_in_test:
                continue

            try:
                user_predictions_df = self.get_user_recommendations(user_id)
                # Prendiamo solo gli indici (movieId) ordinati, fino al max K necessario
                recommended_items_ordered = user_predictions_df.head(max_K).index.tolist()
            except ValueError as e:
                logging.info(f"Errore ottenendo predizioni per user {user_id}: {e}")
                continue
            except Exception as e:  # Cattura altre eccezioni impreviste
                logging.info(f"Eccezione inattesa ottenendo predizioni per user {user_id}: {e}")
                continue

            # 3. Calcola P@k e R@k per ogni k in K_list
            relevant_items_count = len(relevant_items_in_test)
            for k in K_list:
                # Considera solo i primi 'k' elementi raccomandati
                recommended_at_k = set(recommended_items_ordered[:k])
                # L'intersezione ora è tra i primi k e i rilevanti
                true_positives_at_k = len(recommended_at_k.intersection(relevant_items_in_test))

                # Calcola Precision@k
                precision_at_k = true_positives_at_k / k if k > 0 else 0.0

                # Calcola Recall@k
                recall_at_k = true_positives_at_k / relevant_items_count if relevant_items_count > 0 else 0.0

                # Accumula le somme
                precision_sum[k] += precision_at_k
                recall_sum[k] += recall_at_k
                evaluated_user_count[k] += 1  # Incrementa il contatore per questo K

        end_time = time.time()

        logging.info(f"# Valutazione P/R completata in {end_time - start_time:.2f} secondi.")

        # 4. Calcola le medie finali
        results = {}
        logging.info("\n--- Risultati Medi Precision/Recall ---")
        for k in K_list:
            user_count_for_k = evaluated_user_count[k]
            if user_count_for_k > 0:
                avg_precision = precision_sum[k] / user_count_for_k
                avg_recall = recall_sum[k] / user_count_for_k
                results[k] = (avg_precision, avg_recall)
                logging.info(f"  K={k:<3} (Utenti={user_count_for_k}): Precision={avg_precision:.6f}, Recall={avg_recall:.6f}")
            else:
                results[k] = (0.0, 0.0)  # Nessun utente valido per questo K
                logging.info(f"  K={k:<3}: Nessun utente valido per la valutazione.")

        return results
