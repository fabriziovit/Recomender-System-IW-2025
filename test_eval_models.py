import os
import datetime
import pandas as pd
from mf_sgd import MF_SGD_User_Based
from cf_recommender import CollaborativeRecommender
from sklearn.neighbors import NearestNeighbors
from cf_recommender import CollaborativeRecommender
from eval import eval_mae_rmse, eval_precision_recall
from utils import get_train_valid_test_matrix, load_movielens_data, pearson_distance, compute_user_similarity_matrix


# def sgd_eval():
#     # Carica il dataset MovieLens
#     _, df_ratings, _ = load_movielens_data("dataset/")

#     # Crea la utility matrix
#     utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
#     print(f"Dimensioni utility matrix: {utility_matrix.shape}")

#     # Splitting in training e test matrix
#     train_matrix, valid_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index)

#     # Parametri Modello
#     n_epochs: int = 3000
#     num_factors_list: list = [20]  #  [10, 20, 30, 50]  # Test con diversi numeri di fattori latenti
#     learning_rate_list: list = [0.001]  #! [0.001]  # Test con diversi learning rates
#     lambda_list: list = [0.001]  # Test con diversi valori di lambda (weight decay)
#     # lambda_list: list = [1.0, 0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]  #! Test con diversi valori di lambda (weight decay)

#     #
#     # Crea directory per i risultati e i modelli
#     os.makedirs("results", exist_ok=True)
#     os.makedirs("models", exist_ok=True)

#     for n_factor in num_factors_list:
#         for learning_rate in learning_rate_list:
#             for reg in lambda_list:
#                 evaluation_output = []

#                 # 5 Modello Matrix Factorization SGD
#                 recomm = MF_SGD_User_Based(n_factor, learning_rate, reg, n_epochs, utility_matrix, train_matrix, valid_matrix)

#                 # 6. Fit del modello MF su Training
#                 recomm.fit(refit=True, evaluation_output=evaluation_output)

#                 # Predizioni per tutti gli utenti
#                 train_predictions_dict = recomm._predictions_train

#                 mae, rmse = eval_mae_rmse(test_matrix, train_predictions_dict)

#                 evaluation_output.append(f"\### Risultati MAE e RMSE ###")
#                 evaluation_output.append(f"  num_factors: {n_factor}, learning_rate: {learning_rate}, lambda: {reg}")
#                 evaluation_output.append(f"  MAE: {mae:.10f}")
#                 evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

#                 print(f"\### Risultati MAE e RMSE ###")
#                 print(f"  num_factors: {n_factor}, learning_rate: {learning_rate}, lambda: {reg}")
#                 print(f"  MAE: {mae:.10f}")
#                 print(f"  RMSE: {rmse:.10f}\n")

#                 # Salva i risultati su file
#                 model_name = f"mf_model_n{n_factor}_lr{learning_rate}_lambda{reg}_norm"
#                 with open(f"results/{model_name}.txt", "w") as f:
#                     f.write("\n" + "=" * 70 + "\n")
#                     f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
#                     for line in evaluation_output:
#                         f.write(line + "\n")
#                     f.write("=" * 70 + "\n")

#                 # Salva ogni modello
#                 model_path = f"models/{model_name}.pkl"
#                 recomm.save_model(model_path)
#                 print(f"Modello salvato in: {model_path}\n")

# ******************************************************************************************************************** #

# def matrix_knn_eval_cf_user():
#     # Carica il dataset MovieLens
#     df_movies, df_ratings, _ = load_movielens_data("dataset/")

#     # Crea la utility matrix (URM)
#     utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
#     print(f"Dimensioni utility matrix: {utility_matrix.shape}")

#     # Calcola la matrice di similarità utente-utente
#     user_similarity_matrix: pd.DataFrame = compute_user_similarity_matrix(utility_matrix)
#     print(f"user_similarity_matrix: {user_similarity_matrix.shape}")

#     # Splitting in training e test matrix
#     train_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, ret_valid=False)

#     # Crea directory per i risultati e i modelli
#     os.makedirs("results", exist_ok=True)
#     os.makedirs("models", exist_ok=True)
#     evaluation_output: list = []

#     NN_list = [50]  # Numero di vicini da considerare (potrebbe essere ridotto per test)
#     for NN in NN_list:
#         # 5. Inizializza il modello CollaborativeRecommender **passando la matrice e train_matrix**
#         knn_model_pearson_item = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
#         knn_model_pearson_user = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
#         recomm = CollaborativeRecommender(knn_model_pearson_item, knn_model_pearson_user, user_similarity_matrix, utility_matrix)
#         recomm.fit_user_model(train_matrix)
#         recomm.fit_item_model(train_matrix)

#         # Predizioni per il **TEST SET** (corretto!)
#         train_predictions_dict = recomm.compute_predictions_on_train(NN, train_matrix)

#         mae, rmse = eval_mae_rmse(test_matrix, train_predictions_dict)

#         evaluation_output.append(f"\### Risultati MAE e RMSE ###")
#         evaluation_output.append(f"  MAE: {mae:.10f}")
#         evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

#         print(f"\### Risultati MAE e RMSE ###")
#         print(f"  MAE: {mae:.10f}")
#         print(f"  RMSE: {rmse:.10f}\n")

#         # Salva i risultati su file
#         model_name = f"knn_offline_model_NN{NN}_pearson_user_matrix"  # Nome modello più descrittivo
#         with open(f"results/{model_name}.txt", "w") as f:
#             f.write("\n" + "=" * 70 + "\n")
#             f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
#             for line in evaluation_output:
#                 f.write(line + "\n")
#             f.write("=" * 70 + "\n")


# ******************************************************************************************************************** #
# def knn_eval_cf_user():
#     # Carica il dataset MovieLens
#     df_movies, df_ratings, _ = load_movielens_data("dataset/")

#     # Crea la utility matrix (URM)
#     utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
#     print(f"Dimensioni utility matrix: {utility_matrix.shape}")

#     # Splitting in training e test matrix
#     train_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, ret_valid=False)

#     # Crea directory per i risultati e i modelli
#     os.makedirs("results", exist_ok=True)
#     os.makedirs("models", exist_ok=True)
#     evaluation_output: list = []

#     NN_list = [50]  # Numero di vicini da considerare (potrebbe essere ridotto per test)
#     for NN in NN_list:
#         # 5. Inizializza il modello CollaborativeRecommender **passando la matrice e train_matrix**
#         knn_model_pearson_item = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
#         knn_model_pearson_user = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
#         recomm = CollaborativeRecommender(knn_model_pearson_item, knn_model_pearson_user, None, utility_matrix)  # Passa matrice e train_matrix
#         recomm.fit_user_model(train_matrix)
#         recomm.fit_item_model(train_matrix)

#         # Predizioni per il **TEST SET** (corretto!)
#         train_predictions_dict = recomm.knn_compute_predictions_on_train(NN, train_matrix)

#         mae, rmse = eval_mae_rmse(test_matrix, train_predictions_dict)

#         evaluation_output.append(f"\### Risultati MAE e RMSE ###")
#         evaluation_output.append(f"  MAE: {mae:.10f}")
#         evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

#         print(f"\### Risultati MAE e RMSE ###")
#         print(f"  MAE: {mae:.10f}")
#         print(f"  RMSE: {rmse:.10f}\n")

#         # Salva i risultati su file
#         model_name = f"knn_online_model_NN{NN}_pearson_user_matrix"  # Nome modello più descrittivo
#         with open(f"results/{model_name}.txt", "w") as f:
#             f.write("\n" + "=" * 70 + "\n")
#             f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
#             for line in evaluation_output:
#                 f.write(line + "\n")
#             f.write("=" * 70 + "\n")


# ******************************************************************************************************************** #


def knn_eval_precision_recall():

    # Carica il dataset MovieLens
    _, df_ratings, _ = load_movielens_data("dataset/")

    # Crea la utility matrix (URM)
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Dimensioni utility matrix: {utility_matrix.shape}")

    # Splitting in training e test matrix
    train_matrix, test_matrix = get_train_valid_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index, ret_valid=False)

    # # Parametri di Test
    K_list: list[int] = [5, 10, 15, 20]  # Lista di valori per K (numero di raccomandazioni)
    nn_list: list = [5]  # Lista di valori per n_neighbors
    relevant_values: list[float] = [2.0, 2.5, 3.0, 3.5, 4.0]  #! Soglia di rating per considerare un film "rilevante"

    # Per ogni valore in nn_list, crea il modello, il recommender e valuta per tutti i K in K_list
    for NN in nn_list:
        print(f"\n\n### Valutazione per NN = {NN} e per K = {K_list}###")
        knn_model_user = NearestNeighbors(n_neighbors=NN + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        knn_model_item = NearestNeighbors(n_neighbors=NN + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        recomm = CollaborativeRecommender(knn_model_item, knn_model_user)
        recomm.fit_user_model(train_matrix, re_fit=True)
        print(f"# Modello addestrato con n_neighbors = {NN}")

        computed_prediction: dict = recomm.compute_predictions_on_train(NN, train_matrix, exclude=True)
        print(f"###[1 test] type(computed_prediction): {type(computed_prediction)}")
        for user_id, prediction in computed_prediction.items():
            print(f"###[2 test] user_id: {user_id},type(prediction): {type(prediction)}")

        print(f"# Predizioni pre-calcolate per {len(computed_prediction.keys())} utenti")

        for value in relevant_values:
            print(f"# Valutazione per valore {value}...")
            eval_precision_recall(train_matrix, test_matrix, NN, K_list, value, computed_prediction)
            print(f"# Valutazione per valore {value} completata!")


# ******************************************************************************************************************** #

if __name__ == "__main__":
    # eval_cf_user_matrix()
    # knn_eval_cf_user()
    knn_eval_precision_recall()
