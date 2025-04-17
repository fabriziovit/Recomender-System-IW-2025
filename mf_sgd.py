import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MF_SGD_User_Based:

    def __init__(self, num_factors, learning_rate, lambda_reg, n_epochs, utility_matrix, train_matrix, valid_matrix):
        """Inizializza il modello Matrix Factorization con SGD con bias."""
        self.num_factors: int = num_factors
        self.learning_rate: float = learning_rate
        self.lambda_reg: float = lambda_reg
        self.n_epochs: int = n_epochs
        self._is_fitted: bool = False

        self._utility_matrix: pd.DataFrame = utility_matrix

        self._train_samples: list[tuple] = None
        self._train_matrix: pd.DataFrame = train_matrix

        if not utility_matrix.shape[0] == train_matrix.shape[0]:
            raise ValueError("Le matrici utility e train devono avere lo stesso numero di utenti (righe).")
        if not utility_matrix.shape[1] == train_matrix.shape[1]:
            raise ValueError("Le matrici utility e train devono avere lo stesso numero di item (colonne).")

        self._valid_samples: list[tuple] = None
        self._valid_matrix: pd.DataFrame = valid_matrix
        if not utility_matrix.shape[0] == train_matrix.shape[0]:
            raise ValueError("Le matrici utility e valid devono avere lo stesso numero di utenti (righe).")
        if not utility_matrix.shape[1] == train_matrix.shape[1]:
            raise ValueError("Le matrici utility e valid devono avere lo stesso numero di item (colonne).")

        self.X_user_factors: np.ndarray = None  # Matrice X (User) Latent Factors
        self.Y_item_factors: np.ndarray = None  # Matrice Y (Item) Latent Factors

        self.user_biases: np.ndarray = None  # Bias utente
        self.item_biases: np.ndarray = None  # Bias item

        self.user_ids_map: dict = {}  # Mappa user_id -> indice matrice
        self.item_ids_map: dict = {}  # Mappa movie_id -> indice matrice

        self.train_mean: float = None  # Media globale on training set

    def _get_samples(self, matrix) -> list[tuple]:
        """Restituire una lista di samples (user_index, item_index, rating) da una matrice."""
        logging.info("Creazione samples...")
        samples = []
        stacked_matrix = matrix.stack()  # Reshape DataFrame to Series (multi-index)

        # Filter for ratings > 0.0 directly on the Series
        explicit_ratings_series = stacked_matrix[stacked_matrix > 0.0]

        for index, rating in explicit_ratings_series.items():  # Iterate over the filtered Series
            user_id_idx = self.user_ids_map.get(index[0])  # Get user_index from user_id (using map)
            item_id_idx = self.item_ids_map.get(index[1])  # Get item_index from item_id (using map)
            samples.append((user_id_idx, item_id_idx, rating))  # Append sample
        return samples

    def _get_mse_loss_regularized(self, samples) -> float:
        """Calcola la funzione di perdita (MSE con regolarizzazione L2) inclusi i bias e la media globale."""
        user_indices, item_indices, real_ratings = zip(*samples)

        user_indices = np.array(user_indices)
        item_indices = np.array(item_indices)
        real_ratings = np.array(real_ratings)

        predicted_ratings = (
            self.train_mean
            + self.user_biases[user_indices]
            + self.item_biases[item_indices]
            + np.sum(self.X_user_factors[user_indices] * self.Y_item_factors[item_indices], axis=1)
        )

        partial_res = 0.5 * self.lambda_reg * (np.sum(self.X_user_factors**2) + np.sum(self.Y_item_factors**2) + np.sum(self.user_biases**2) + np.sum(self.item_biases**2))
        errors = real_ratings - predicted_ratings
        mse_loss = 0.5 * np.sum(errors**2)
        return mse_loss + partial_res

    def _predict_rating(self, user_index: int, item_index: int) -> float:
        """Predice il rating con bias e media globale (de-normalizzazione)."""
        return self.train_mean + self.user_biases[user_index] + self.item_biases[item_index] + np.dot(self.X_user_factors[user_index], self.Y_item_factors[item_index])

    def _get_predictions(self, user_id: int, matrix: pd.DataFrame, exclude: bool = True) -> pd.DataFrame:
        """Restituisce le predizioni per un utente specifico."""
        if user_id not in self.user_ids_map:
            raise ValueError(f"User ID {user_id} non trovato nel training set.")
        predicted_ratings_list = []
        item_ids = matrix.columns  # All movie ids from the matrix
        user_index = self.user_ids_map[user_id]
        seen_movies_ids = matrix.loc[user_id][matrix.loc[user_id] > 0].index.tolist()  # Film visti dall'utente
        for item_id in item_ids:
            if exclude and item_id in seen_movies_ids:
                continue  # Salta i film già visti se exclue è True
            item_index = self.item_ids_map[item_id]
            predicted_rating = self._predict_rating(user_index, item_index)
            predicted_ratings_list.append({"movieId": item_id, "predicted_rating": predicted_rating})
        predictions_df = pd.DataFrame(predicted_ratings_list).set_index("movieId")
        predictions_df.columns = ["values"]
        return predictions_df.sort_values(by="values", ascending=False)

    def _stochastic_gradient_descent(self, num_factors: int, learning_rate: float, lambda_term: float, refit: bool = False, evaluation_output: list = None) -> None:
        if not self._is_fitted or refit:
            logging.info("Effettuo l'addestramento con bias e media globale...")
            logging.info(f"Iperparametri: num_factors={num_factors}, learning_rate={learning_rate}, lambda={lambda_term}")  # Log degli iperparametri
            best_valid_loss_reg = float("inf")
            best_epoch = 0
            patience = 10
            epochs_no_improve = 0

            best_user_factors = None
            best_item_factors = None
            best_user_biases = None
            best_item_biases = None

            for epoch in range(0, self.n_epochs):
                loss_mse = 0
                np.random.shuffle(self._train_samples)
                for user_index, item_index, real_rating in self._train_samples:
                    # 0. Calcolo errore
                    predicted_rating = self._predict_rating(user_index, item_index)
                    error_ij = real_rating - predicted_rating
                    # 1. Recupero dei fattori latenti e dei bias
                    user_factor = self.X_user_factors[user_index]  # Fattori latenti utente
                    item_factor = self.Y_item_factors[item_index]  # Fattori latenti item
                    user_bias = self.user_biases[user_index]  # Bias utente
                    item_bias = self.item_biases[item_index]  # Bias item
                    # 2. Aggiornamento dei fattori latenti e dei bias con calcolo dei gradienti
                    user_gradient = -(error_ij * item_factor) + (lambda_term * user_factor)  # Gradiente per fattori utente
                    item_gradient = -(error_ij * user_factor) + (lambda_term * item_factor)  # Gradiente per fattori item
                    user_bias_gradient = -error_ij + (lambda_term * user_bias)  # Gradiente per bias utente
                    item_bias_gradient = -error_ij + (lambda_term * item_bias)  # Gradiente per bias item
                    # 4. Aggiornamento dei fattori latenti e dei bias
                    self.X_user_factors[user_index] -= self.learning_rate * user_gradient  # Aggiornamento fattori utente
                    self.Y_item_factors[item_index] -= self.learning_rate * item_gradient  # Aggiornamento fattori item
                    self.user_biases[user_index] -= self.learning_rate * user_bias_gradient  # Aggiornamento bias utente
                    self.item_biases[item_index] -= self.learning_rate * item_bias_gradient  # Aggiornamento bias item
                    # 5. Somma dell'errore quadratico
                    loss_mse += error_ij**2
                # 6. Calcolo dell'errore quadratico medio (MSE)
                loss_mse /= len(self._train_samples)
                # 7. Calcolo dell'errore quadratico medio regolarizzato (MSE + Weight Decay (L2))
                train_loss_reg = self._get_mse_loss_regularized(self._train_samples)
                valid_loss_reg = self._get_mse_loss_regularized(self._valid_samples)

                log_msg = f"Epoch {epoch+1}/{self.n_epochs}, Train-Loss: {loss_mse:.10f}, Train-Loss-Reg: {train_loss_reg:.10f}, Valid-Loss-Reg: {valid_loss_reg:.10f}, K: {num_factors}, lrate: {learning_rate}, lambda: {lambda_term}"
                logging.info(log_msg)
                if evaluation_output is not None:
                    evaluation_output.append(log_msg)

                # Early stopping criteria
                if valid_loss_reg < best_valid_loss_reg:
                    best_valid_loss_reg = valid_loss_reg
                    best_epoch = epoch + 1
                    epochs_no_improve = 0

                    best_user_factors = self.X_user_factors.copy()
                    best_item_factors = self.Y_item_factors.copy()

                    best_user_biases = self.user_biases.copy()
                    best_item_biases = self.item_biases.copy()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logging.info(f"Early stopping attivato all'epoca {epoch+1}!")
                        logging.info(f"Valid-Loss-Reg non migliorata per {patience} epoche.")
                        log_msg = f"Miglior epoca: {best_epoch} con Train-Loss: {loss_mse:.10f}, Train-Loss-Reg:{train_loss_reg:.10f},  Valid-Loss-Reg: {best_valid_loss_reg:.10f}"
                        logging.info(log_msg)
                        if evaluation_output is not None:
                            evaluation_output.append(log_msg)
                        # Ripristino i migliori parametri
                        self.X_user_factors = best_user_factors
                        self.Y_item_factors = best_item_factors
                        self.user_biases = best_user_biases
                        self.item_biases = best_item_biases
                        return

    def fit(self, refit: bool = False, evaluation_output: list = None) -> None:

        user_ids = self._train_matrix.index
        item_ids = self._train_matrix.columns

        # Mapping user_id e item_id a indici matrice
        self.user_ids_map = {user_id: index for index, user_id in enumerate(user_ids)}
        self.item_ids_map = {item_id: index for index, item_id in enumerate(item_ids)}

        # Get Simples for train and test
        self._train_samples = self._get_samples(self._train_matrix)
        self._valid_samples = self._get_samples(self._valid_matrix)
        logging.info(f"train_simples: {len(self._train_samples)}, valid_simples: {len(self._valid_samples)}")

        # Inizializzazione dei fattori latenti e dei bias
        self.X_user_factors = np.random.normal(0, 0.01, (len(user_ids), self.num_factors))
        self.Y_item_factors = np.random.normal(0, 0.01, (len(item_ids), self.num_factors))
        self.user_biases = np.zeros(len(user_ids), dtype=np.float64)
        self.item_biases = np.zeros(len(item_ids), dtype=np.float64)

        # Calcolo della media globale sul training set per la normalizzazione
        self.train_mean = self._train_matrix[self._train_matrix > 0.0].stack().mean()
        logging.info(f"Media globale sul training set: {self.train_mean:.10f}")

        # Addestramento del modello
        self._stochastic_gradient_descent(self.num_factors, self.learning_rate, self.lambda_reg, refit, evaluation_output)
        self._is_fitted = True

    def get_recommendations(self, utility_matrix: pd.DataFrame, user_id: pd.Index) -> pd.DataFrame:
        """Restituisce le predizioni per l'utente specificato."""
        # Exclude i film già visti dall'utente
        recomm_df = self._get_predictions(user_id, utility_matrix, exclude=True)
        return recomm_df

    @classmethod
    def load_model(cls, filepath):
        """Carica un modello precedentemente salvato da disco."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File del modello non trovato: {filepath}")
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model

    def save_model(self, filepath):
        """Salva il modello addestrato su disco usando pickle."""
        if not self._is_fitted:
            raise ValueError("Il modello deve essere prima addestrato (fit) prima di poterlo salvare.")
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logging.info(f"Modello salvato con successo in: {filepath}")

    # **************************************************************************** #

    def evaluate_mea_rmse(self, test_matrix: pd.DataFrame, min_rating: float = 0.5, max_rating: float = 5.0):
        """Valuta il modello sul test set calcolando MAE e RMSE."""
        if not self._is_fitted:
            raise RuntimeError("Il modello MF deve essere addestrato prima della valutazione. Chiamare fit().")

        logging.info("Inizio valutazione MAE/RMSE...")

        true_ratings = []
        predicted_ratings = []
        skipped_count = 0
        processed_count = 0
        start_time = time.time()

        # Iterazione Efficiente sul Test Set usando stack()
        test_ratings_series = test_matrix[test_matrix > 0].stack()
        total_ratings_to_predict = len(test_ratings_series)

        if total_ratings_to_predict == 0:
            logging.warning("Nessun rating > 0 trovato nel test_matrix fornito per la valutazione.")
            return np.nan, np.nan

        logging.info(f"Valutazione su {total_ratings_to_predict} rating nel test set...")

        # Iteriamo sulla Series risultante (indice multi-livello user_id, movie_id)
        for idx, true_rating in test_ratings_series.items():
            user_id, movie_id = idx  # Estrai userId e movieId dall'indice

            processed_count += 1

            # Ottieni gli indici interni corrispondenti agli ID
            user_idx = self.user_ids_map.get(user_id)
            item_idx = self.item_ids_map.get(movie_id)

            # Salta se l'utente o l'item non erano presenti nel training set
            # Il modello non può fare predizioni per dati mai visti durante il training
            if user_idx is None or item_idx is None:
                skipped_count += 1
                continue

            # Calcola la predizione usando gli indici interni
            try:
                predicted_rating_raw = self._predict_rating(user_idx, item_idx)

                # Applica il clipping per riportare la predizione nel range valido
                predicted_rating_clipped = np.clip(predicted_rating_raw, min_rating, max_rating)

                # Aggiungi i valori alle liste
                true_ratings.append(true_rating)
                predicted_ratings.append(predicted_rating_clipped)

            except IndexError:
                # Questo potrebbe accadere se le mappe o i fattori non sono coerenti
                logging.error(f"Indice fuori dai limiti per ({user_id},{movie_id}) -> ({user_idx},{item_idx}). Skipping.")
                skipped_count += 1
                continue
            except Exception as e:
                logging.error(f"Errore generico durante la predizione per ({user_id},{movie_id}): {e}")
                skipped_count += 1
                continue

        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f"Valutazione MAE/RMSE completata in {elapsed_time:.2f} secondi.")
        if skipped_count > 0:
            logging.info(f"  Skipped {skipped_count} rating (user/item non nel training o errore).")

        # Calcola MAE e RMSE se ci sono state predizioni valide
        if not true_ratings:
            logging.warning("Nessuna predizione valida generata per il calcolo di MAE/RMSE.")
            return np.nan, np.nan
        else:
            mae = mean_absolute_error(true_ratings, predicted_ratings)
            rmse = root_mean_squared_error(true_ratings, predicted_ratings)  # squared=False per RMSE
            logging.info(f"  MAE:  {mae:.10f}")
            logging.info(f"  RMSE: {rmse:.10f}")
            return mae, rmse

    # **************************************************************************** #

    def evaluate_precision_recall(self, test_matrix: pd.DataFrame, K_list: list[int], relevant_threshold: float):
        """Valuta il modello sul test set calcolando Precision@K e Recall@K."""

        if not self._is_fitted:
            raise RuntimeError("Il modello MF deve essere addestrato prima della valutazione P/R. Chiamare fit().")

        # Dizionari per accumulare somme e conteggi
        precision_sum = defaultdict(float)
        recall_sum = defaultdict(float)
        evaluated_user_count = defaultdict(int)  # Contatore utenti validi per ogni K
        max_K = max(K_list) if K_list else 0
        processed_users = 0
        start_time = time.time()

        if max_K == 0:
            logging.warning("K_list è vuota. Nessuna metrica P/R calcolata.")
            return {}

        # Identifica utenti presenti sia nel test set che nel training set (mappa ID)
        test_users = test_matrix.index
        valid_test_users = test_users.intersection(self.user_ids_map.keys())
        total_users_to_evaluate = len(valid_test_users)

        if total_users_to_evaluate == 0:
            logging.warning("Nessun utente in comune tra test_matrix e training set.")
            return {k: (0.0, 0.0) for k in K_list}

        logging.info(f"Inizio valutazione Precision/Recall per K={K_list}, soglia={relevant_threshold}...")
        logging.info(f"Valutazione su {total_users_to_evaluate} utenti comuni.")

        # Itera sugli utenti validi
        for user_id in valid_test_users:
            processed_users += 1

            # 1. Trova gli item rilevanti per l'utente nel test set
            test_user_ratings = test_matrix.loc[user_id]
            relevant_items_in_test = set(test_user_ratings[test_user_ratings >= relevant_threshold].index)
            relevant_items_count = len(relevant_items_in_test)

            # Se l'utente non ha item rilevanti nel test set, salta
            if relevant_items_count == 0:
                continue

            # 2. Ottieni le raccomandazioni TOP K (ordinate) per l'utente
            try:
                user_predictions_df = self._get_predictions(user_id, self._train_matrix, exclude=True)
                # Prendiamo solo gli indici (movieId) ordinati, fino al max K necessario
                recommended_items_ordered = user_predictions_df.head(max_K).index.tolist()
            except ValueError:  # Potrebbe accadere se l'utente non è nella mappa (anche se filtrato)
                logging.warning(f"User ID {user_id} non trovato durante _get_predictions (imprevisto). Skipping.")
                continue
            except Exception as e:
                logging.error(f"Errore inatteso ottenendo predizioni P/R per user {user_id}: {e}")
                continue

            # 3. Calcola P@k e R@k per ogni k in K_list
            for k in K_list:
                if k <= 0:
                    continue  # Salta K non validi

                # Considera solo i primi 'k' elementi raccomandati
                recommended_at_k = set(recommended_items_ordered[:k])

                # Calcola True Positives come intersezione tra i primi k raccomandati e i rilevanti
                true_positives_at_k = len(recommended_at_k.intersection(relevant_items_in_test))

                # Calcola Precision@k
                precision_at_k = true_positives_at_k / k

                # Calcola Recall@k
                # La divisione per relevant_items_count è sicura perché abbiamo controllato che sia > 0
                recall_at_k = true_positives_at_k / relevant_items_count

                # Accumula le somme per le medie finali
                precision_sum[k] += precision_at_k
                recall_sum[k] += recall_at_k
                evaluated_user_count[k] += 1  # Incrementa il contatore SOLO se l'utente è stato valutato per questo K

        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f"Valutazione Precision/Recall completata in {elapsed_time:.2f} secondi.")

        # 4. Calcola le medie finali
        results = {}

        logging.info("--- Risultati Medi Precision/Recall ---")
        for k in K_list:
            if k <= 0:
                continue
            user_count_for_k = evaluated_user_count[k]
            if user_count_for_k > 0:
                avg_precision = precision_sum[k] / user_count_for_k
                avg_recall = recall_sum[k] / user_count_for_k
                results[k] = (avg_precision, avg_recall)
                logging.info(f"  K={k:<3} (Utenti={user_count_for_k}): Precision={avg_precision:.6f}, Recall={avg_recall:.6f}")
            else:
                # Se nessun utente è stato valutato per questo K (improbabile se total_users > 0 ma possibile)
                results[k] = (0.0, 0.0)
                logging.info(f"  K={k:<3}: Nessun utente valutato.")
        return results
