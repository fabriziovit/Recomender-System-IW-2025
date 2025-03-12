import os
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
from eval import compute_mae_rmse
from utils import load_movielens_data, get_train_valid_test_matrix

# pd.set_option("display.max_rows", None)  # Non limitare il numero di righe
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ************************************************************************************************************** #


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

        self._predictions_train: dict = None  # Predizioni per tutti gli utenti

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

    def _normalize_matrix(self, matrix) -> pd.DataFrame:
        """Normalizza sottraendo la media globale."""
        return matrix - self.train_mean

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
        """Raccomanda film per un utente dato, basato sul modello Matrix Factorization SGD."""
        if user_id not in self.user_ids_map:
            raise ValueError(f"User ID {user_id} non trovato nel training set.")
        predicted_ratings_list = []
        user_index = self.user_ids_map[user_id]
        item_ids = matrix.columns  # All movie ids from the matrix
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

    def _compute_predictions_on_train(self) -> dict:
        """Calcola le predizioni sul training set"""
        all_user_predictions_evaluation = {}
        for user_id in self._train_matrix.index:
            all_user_predictions_evaluation[user_id] = self._get_predictions(user_id, self._train_matrix, exclude=False)
        return all_user_predictions_evaluation

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
                        self._is_fitted = True
                        return
            self._is_fitted = True

    def fit(self, refit: bool = False, evaluation_output: list = None) -> None:

        user_ids = self._train_matrix.index
        item_ids = self._train_matrix.columns

        # Mapping user_id e item_id a indici matrice
        self.user_ids_map = {user_id: index for index, user_id in enumerate(user_ids)}
        self.item_ids_map = {item_id: index for index, item_id in enumerate(item_ids)}

        # Get Simples for train and test
        self._train_samples = self._get_samples(self._train_matrix)
        self._valid_samples = self._get_samples(self._valid_matrix)
        print(f"train_simples: {len(self._train_samples)}, valid_simples: {len(self._valid_samples)}")

        # Inizializzazione dei fattori latenti e dei bias
        self.X_user_factors = np.random.normal(0, 0.01, (len(user_ids), self.num_factors))
        self.Y_item_factors = np.random.normal(0, 0.01, (len(item_ids), self.num_factors))
        self.user_biases = np.zeros(len(user_ids), dtype=np.float64)
        self.item_biases = np.zeros(len(item_ids), dtype=np.float64)

        # Calcolo della media globale sul training set per la normalizzazione
        self.train_mean = self._train_matrix[self._train_matrix > 0.0].stack().mean()
        print(f"Media globale sul training set: {self.train_mean:.10f}")

        # Normalizzazione della matrice di training, validazione e test
        self._train_matrix = self._normalize_matrix(self._train_matrix)
        self._valid_matrix = self._normalize_matrix(self._valid_matrix)
        print("Matrici normalizzate con successo.")

        # Addestramento del modello
        self._stochastic_gradient_descent(self.num_factors, self.learning_rate, self.lambda_reg, refit, evaluation_output)

        # Calcolo delle predizioni per tutti gli utenti
        self._predictions_train = self._compute_predictions_on_train()
        print("Predizioni per tutti gli utenti calcolate con successo.")

    def get_recommendations(self, matrix: pd.DataFrame, user_id: pd.Index) -> pd.DataFrame:
        """Restituisce le predizioni per l'utente specificato."""
        return self._get_predictions(user_id, matrix, exclude=True)

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
        print(f"Modello salvato con successo in: {filepath}")


# ************************************************************************************************************** #


def eval():
    # 1. Carica il dataset MovieLens
    _, df_ratings, _ = load_movielens_data("dataset/")

    # 2. Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"Dimensioni utility matrix: {utility_matrix.shape}")

    # 3. Splitting in training e test matrix
    train_matrix, valid_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index)

    # Parametri Modello
    n_epochs: int = 3000
    num_factors_list: list = [500]  #  [10, 20, 30, 50]  # Test con diversi numeri di fattori latenti
    learning_rate_list: list = [0.001]  #! [0.001]  # Test con diversi learning rates
    # lambda_list: list = [0.001, 0.0001, 0.00001]  # Test con diversi valori di lambda (weight decay)
    lambda_list: list = [1.0, 0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]  #! Test con diversi valori di lambda (weight decay)

    #
    # Crea directory per i risultati e i modelli
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for n_factor in num_factors_list:
        for learning_rate in learning_rate_list:
            for reg in lambda_list:
                evaluation_output = []

                # 5 Modello Matrix Factorization SGD
                recomm = MF_SGD_User_Based(n_factor, learning_rate, reg, n_epochs, utility_matrix, train_matrix, valid_matrix)

                # 6. Fit del modello MF su Training
                recomm.fit(refit=True, evaluation_output=evaluation_output)

                # Predizioni per tutti gli utenti
                train_predictions_dict = recomm._predictions_train

                mae, rmse = compute_mae_rmse(test_matrix, train_predictions_dict)

                evaluation_output.append(f"\### Risultati MAE e RMSE ###")
                evaluation_output.append(f"  num_factors: {n_factor}, learning_rate: {learning_rate}, lambda: {reg}")
                evaluation_output.append(f"  MAE: {mae:.10f}")
                evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

                print(f"\### Risultati MAE e RMSE ###")
                print(f"  num_factors: {n_factor}, learning_rate: {learning_rate}, lambda: {reg}")
                print(f"  MAE: {mae:.10f}")
                print(f"  RMSE: {rmse:.10f}\n")

                # Salva i risultati su file
                model_name = f"mf_model_n{n_factor}_lr{learning_rate}_lambda{reg}_norm"
                with open(f"results/{model_name}.txt", "w") as f:
                    f.write("\n" + "=" * 70 + "\n")
                    f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
                    for line in evaluation_output:
                        f.write(line + "\n")
                    f.write("=" * 70 + "\n")

                # Salva ogni modello
                model_path = f"models/{model_name}.pkl"
                recomm.save_model(model_path)


# ************************************************************************************************************** #

if __name__ == "__main__":
    # ########################################################### #
    eval()
    # ########################################################### #
    # 1. Carica il dataset MovieLens
    # df_movies, df_ratings, df_tags = load_movielens_data("dataset/")

    # # 2. Crea la utility matrix
    # utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # user_ids_to_test = [100, 604]

    # model_path1 = "./models/mf_model_n70_lr0.001_lambda1e-05_norm.pkl"
    # recomm1: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path1)

    # model_path2 = "./models/mf_model_n140_lr0.001_lambda1e-05_norm.pkl"
    # recomm2: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path2)

    # for user_id in user_ids_to_test:
    #     print(f"\n\nUser ID: {user_id}")
    #     print(recomm2.get_recommendations(utility_matrix, user_id).head(5).merge(df_movies, on="movieId")[["title", "values"]])
