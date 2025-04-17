import os
import logging
import datetime
from mf_sgd import MF_SGD_User_Based
from sklearn.neighbors import NearestNeighbors
from cf_recommender import CollaborativeRecommender
from cf_recommender import CollaborativeRecommender
from utils import get_train_valid_test_matrix, load_movielens_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ******************************************************************************************************************** #
def cf_run_eval_mae_rmse():
    # Carica il dataset MovieLens
    _, df_ratings, _ = load_movielens_data("dataset/")

    # Crea la utility matrix (URM)
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    logging.info(f"Dimensioni utility matrix: {utility_matrix.shape}")

    # Splitting in training e test matrix
    train_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, ret_valid=False)

    logging.info(f"Dimensioni train_matrix: {train_matrix.shape}")
    logging.info(f"Dimensioni test_matrix: {test_matrix.shape}")
    logging.info(f"Training.head() :\n {train_matrix.head()}")
    logging.info(f"Test.head() :\n {test_matrix.head()}")

    # Crea directory per i risultati e i modelli
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    evaluation_output: list = []

    NN_list = [600]  # Numero di vicini da considerare (potrebbe essere ridotto per test)
    for NN in NN_list:
        # 5. Inizializza il modello CollaborativeRecommender **passando la matrice e train_matrix**
        knn_model_pearson_item = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
        knn_model_pearson_user = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
        recomm = CollaborativeRecommender(knn_model_pearson_item, knn_model_pearson_user)  # Passa matrice e train_matrix
        recomm.fit_user_model(train_matrix)

        mae, rmse = recomm.evaluate_mae_rmse(test_matrix)

        evaluation_output.append(f"\### Risultati MAE e RMSE ###")
        evaluation_output.append(f"  MAE: {mae:.10f}")
        evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

        logging.info(f"\### Risultati MAE e RMSE ###")
        logging.info(f"  MAE: {mae:.10f}")
        logging.info(f"  RMSE: {rmse:.10f}\n")

        # Salva i risultati su file
        model_name = f"new_knn_model_NN{NN}"  # Nome modello più descrittivo
        with open(f"results/{model_name}.txt", "w") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
            for line in evaluation_output:
                f.write(line + "\n")
            f.write("=" * 70 + "\n")


# def cf_run_eval_precision_recall():
#     # Carica il dataset MovieLens
#     logging.info("Caricamento dati...")
#     # df_movies, df_ratings, _ = load_movielens_data("dataset/") # Carica anche df_movies se serve per info
#     _, df_ratings, _ = load_movielens_data("dataset/")  # Assicurati path corretto
#     # Crea la utility matrix (URM)
#     logging.info("Creazione utility matrix...")
#     utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
#     logging.info(f"Dimensioni utility matrix: {utility_matrix.shape}")

#     # Splitting in training e test matrix
#     logging.info("Splitting in training e test set...")
#     train_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, ret_valid=False)

#     logging.info(f"Dimensioni train_matrix: {train_matrix.shape}")
#     logging.info(f"Dimensioni test_matrix: {test_matrix.shape}")

#     # Crea directory per i risultati
#     os.makedirs("results_pr", exist_ok=True)  # Usa una directory diversa se preferisci

#     # --- Parametri di Test ---
#     K_LIST: list[int] = [5, 10, 15, 20]  # Lista di valori per K
#     NN_LIST: list = [10]  # Lista di valori per n_neighbors (puoi aggiungere altri)
#     RELEVANT_THRESHOLDS: list[float] = [2.0, 2.5, 3.0, 3.5, 4.0]  # Soglie di rating per rilevanza

#     # --- Loop Principale ---
#     for nn in NN_LIST:
#         logging.info(f"\n{'='*20} Valutazione P/R per NN = {nn} {'='*20}")

#         # 1. Inizializza modelli e recommender (una volta per NN)
#         logging.info("Inizializzazione modelli KNN...")
#         knn_model_item = NearestNeighbors(n_neighbors=nn + 1, metric="cosine", algorithm="brute", n_jobs=-1)
#         knn_model_user = NearestNeighbors(n_neighbors=nn + 1, metric="cosine", algorithm="brute", n_jobs=-1)
#         recomm = CollaborativeRecommender(knn_model_item, knn_model_user)

#         # 2. Addestra il modello sul training set
#         logging.info("Addestramento modello User-based...")
#         recomm.fit_user_model(train_matrix)  # Non serve re_fit=True qui nel loop NN

#         # 3. Loop sulle soglie di rilevanza
#         for threshold in RELEVANT_THRESHOLDS:
#             logging.info(f"\n--- Valutazione per Soglia di Rilevanza = {threshold} ---")

#             # 4. Chiama il metodo di valutazione P/R della classe
#             pr_results = recomm.evaluate_precision_recall(test_matrix, K_LIST, threshold)  # Metti a False per meno output
#             # 5. Salva i risultati
#             model_name = f"new_knn_{threshold}_NN{nn}"
#             results_filename = f"results_pr/{model_name}.txt"
#             logging.info(f"Salvataggio risultati P/R in: {results_filename}")

#             with open(results_filename, "w") as f:
#                 f.write("\n" + "=" * 70 + "\n")
#                 f.write(f"Precision/Recall Evaluation Date: {datetime.datetime.now()}\n")
#                 f.write(f"Model Type: UserBased KNN (Cosine on Mean-Centered)\n")
#                 f.write(f"Number of Neighbors (NN): {nn}\n")
#                 f.write(f"Relevance Threshold: >= {threshold}\n")
#                 f.write(f"K values evaluated: {K_LIST}\n")
#                 f.write("-" * 70 + "\n")
#                 f.write("Average Results:\n")
#                 # Scrivi i risultati medi per ogni K
#                 for k, (precision, recall) in sorted(pr_results.items()):
#                     f.write(f"  K={k:<3}: Precision={precision:.6f}, Recall={recall:.6f}\n")
#                 f.write("=" * 70 + "\n")

#         logging.info("-" * 50)

#     logging.info("\nValutazione Precision/Recall completata per tutti i parametri.")


# ******************************************************************************************************************** #


# def sgd_run_mae_rmse():
#     # Carica il dataset MovieLens
#     _, df_ratings, _ = load_movielens_data("dataset/")

#     # Crea la utility matrix
#     utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
#     logging.info(f"Dimensioni utility matrix: {utility_matrix.shape}")

#     # Splitting in training e test matrix
#     train_matrix, valid_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, True)

#     # Parametri Modello
#     n_epochs: int = 5000
#     num_factors_list: list = [200]
#     learning_rate_list: list = [0.0005, 0.0001]  # [0.001]
#     lambda_list: list = [0.0, 0.001, 0.0001]  # Test con diversi valori di lambda (weight decay)

#     # Crea directory per i risultati e i modelli
#     os.makedirs("results", exist_ok=True)
#     os.makedirs("models", exist_ok=True)

#     for n_factor in num_factors_list:
#         for learning_rate in learning_rate_list:
#             for reg in lambda_list:
#                 evaluation_output = []

#                 # 5 Modello Matrix Factorization SGD
#                 recomm = MatrixFactorizationSGD(n_factor, learning_rate, reg, n_epochs, utility_matrix, train_matrix, valid_matrix)

#                 # 6. Fit del modello MF su Training
#                 recomm.fit(refit=True, evaluation_output=evaluation_output)

#                 mae, rmse = recomm.evaluate_mea_rmse(test_matrix)

#                 evaluation_output.append(f"\### Risultati MAE e RMSE ###")
#                 evaluation_output.append(f"  num_factors: {n_factor}, learning_rate: {learning_rate}, lambda: {reg}")
#                 evaluation_output.append(f"  MAE: {mae:.10f}")
#                 evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

#                 logging.info(f"\### Risultati MAE e RMSE ###")
#                 logging.info(f"  num_factors: {n_factor}, learning_rate: {learning_rate}, lambda: {reg}")
#                 logging.info(f"  MAE: {mae:.10f}")
#                 logging.info(f"  RMSE: {rmse:.10f}\n")

#                 # Salva i risultati su file
#                 model_name = f"new_mf_model_n{n_factor}_lr{learning_rate}_lambda{reg}_norm"
#                 with open(f"results/{model_name}.txt", "w") as f:
#                     f.write("\n" + "=" * 70 + "\n")
#                     f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
#                     for line in evaluation_output:
#                         f.write(line + "\n")
#                     f.write("=" * 70 + "\n")

#                 # Salva ogni modello
#                 model_path = f"models/{model_name}.pkl"
#                 recomm.save_model(model_path)
#                 logging.info(f"Modello salvato in: {model_path}\n")


# def sgd_run_precision_recall():
#     # Carica il dataset MovieLens
#     _, df_ratings, _ = load_movielens_data("dataset/")

#     # Crea la utility matrix
#     utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
#     logging.info(f"Dimensioni utility matrix: {utility_matrix.shape}")

#     # Splitting in training e test matrix
#     train_matrix, valid_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, True)

#     # Parametri Modello
#     n_epochs: int = 5000
#     num_factors_list: list = [20, 30, 40, 50, 70, 80, 100, 120, 140, 160, 180, 200]  # [10, 20, 30, 50]  # Test con diversi numeri di fattori latenti
#     learning_rate_list: list = [0.001]  # [0.001]
#     lambda_list: list = [0.0, 0.001, 0.0001]  # Test con diversi valori di lambda (weight decay)

#     # Crea directory per i risultati e i modelli
#     os.makedirs("mf_pr_results", exist_ok=True)

#     # --- Parametri di Test ---
#     K_LIST: list[int] = [5, 10, 15, 20]  # Lista di valori per K
#     RELEVANT_THRESHOLDS: list[float] = [2.0, 2.5, 3.0, 3.5, 4.0]  # Soglie di rating per rilevanza

#     for n_factor in num_factors_list:
#         for learning_rate in learning_rate_list:
#             for reg in lambda_list:
#                 # recomm = MF_SGD_User_Based.load_model("models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl")
#                 recomm = MF_SGD_User_Based.load_model(f"models/new_mf_model_n{n_factor}_lr{learning_rate}_lambda{reg}_norm.pkl")

#                 # 3. Loop sulle soglie di rilevanza
#                 for threshold in RELEVANT_THRESHOLDS:
#                     logging.info(f"\n--- Valutazione per Soglia di Rilevanza = {threshold} ---")

#                     # 4. Chiama il metodo di valutazione P/R della classe
#                     pr_results = recomm.evaluate_precision_recall(test_matrix, K_LIST, threshold)  # Metti a False per meno output
#                     # 5. Salva i risultati
#                     model_name = f"new_mf_model_n{n_factor}_lr{learning_rate}_lambda{reg}_{threshold}"
#                     results_filename = f"mf_pr_results/{model_name}.txt"
#                     logging.info(f"Salvataggio risultati P/R in: {results_filename}")

#                     with open(results_filename, "w") as f:
#                         f.write("\n" + "=" * 70 + "\n")
#                         f.write(f"Precision/Recall Evaluation Date: {datetime.datetime.now()}\n")
#                         f.write(f"Model Type: UserBased KNN (Cosine on Mean-Centered)\n")
#                         f.write(f"Relevance Threshold: >= {threshold}\n")
#                         f.write(f"K values evaluated: {K_LIST}\n")
#                         f.write("-" * 70 + "\n")
#                         f.write("Average Results:\n")
#                         # Scrivi i risultati medi per ogni K
#                         for k, (precision, recall) in sorted(pr_results.items()):
#                             f.write(f"  K={k:<3}: Precision={precision:.6f}, Recall={recall:.6f}\n")
#                         f.write("=" * 70 + "\n")

#                 logging.info("-" * 50)

#             logging.info("\nValutazione Precision/Recall completata per tutti i parametri.")


# ******************************************************************************************************************** #

if __name__ == "__main__":
    # ******************************************
    #! Collaborative Filtering (CF) - Valutazione
    # Per la valutazione di RMSE e MAE
    cf_run_eval_mae_rmse()
    # Per la valutazione di Precision e Recall
    # cf_run_eval_precision_recall()
    # ******************************************
    #! SGD - Valutazione
    # Per la valutazione di RMSE e MAE
    # sgd_run_mae_rmse()
    # Per la valutazione di Precision e Recall
    # sgd_run_precision_recall()
