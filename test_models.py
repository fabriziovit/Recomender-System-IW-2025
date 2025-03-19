import datetime
import os
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from eval import compute_mae_rmse
from mf_sgd import MF_SGD_User_Based
from cf_recommender import CollaborativeRecommender
from utils import get_train_test_matrix, load_movielens_data, pearson_distance


def compute_user_similarity_matrix(train_matrix, metric=pearson_distance):
    """Calcola la matrice di similarità utente-utente."""
    similarity_matrix = 1 - pairwise_distances(train_matrix, metric=metric, n_jobs=-1)  # 1 - distance perché vogliamo similarità
    similarity_df = pd.DataFrame(similarity_matrix, index=train_matrix.index, columns=train_matrix.index)
    return similarity_df


# def eval_sgd():
#     # 1. Carica il dataset MovieLens
#     _, df_ratings, _ = load_movielens_data("dataset/")

#     # 2. Crea la utility matrix
#     utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
#     print(f"Dimensioni utility matrix: {utility_matrix.shape}")

#     # 3. Splitting in training e test matrix
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

#                 mae, rmse = compute_mae_rmse(test_matrix, train_predictions_dict)

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


def eval_cf_user():
    # 1. Carica il dataset MovieLens
    df_movies, df_ratings, _ = load_movielens_data("dataset/")

    # 2. Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"Dimensioni utility matrix: {utility_matrix.shape}")

    # 3. Splitting in training e test matrix
    train_matrix, valid_matrix, test_matrix = get_train_test_matrix(df_ratings, utility_matrix.columns, utility_matrix.index)

    # Crea directory per i risultati e i modelli
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    NN = len(train_matrix.index)  # Numero di vicini da considerare (potrebbe essere ridotto per test)
    evaluation_output = []

    # 4. Calcola la matrice di similarità utente-utente **UNA SOLA VOLTA**
    user_similarity_matrix = compute_user_similarity_matrix(train_matrix)

    # 5. Inizializza il modello CollaborativeRecommender **passando la matrice e train_matrix**
    knn_model_pearson_item = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN, n_jobs=-1)  # Potresti non usarlo se ti concentri solo su user-based
    recomm = CollaborativeRecommender(knn_model_pearson_item, user_similarity_matrix=user_similarity_matrix, train_matrix=train_matrix)  # Passa matrice e train_matrix
    recomm.fit_user_model(train_matrix, re_fit=True)  # Non necessario
    recomm.fit_item_model(train_matrix, re_fit=True)  # Forse non necessario se user-based

    # Predizioni per il **TEST SET** (corretto!)
    test_predictions_dict = {}
    for user_id in test_matrix.index:
        for movie_id in test_matrix.columns:
            if test_matrix.loc[user_id, movie_id] > 0:  # Predici solo dove ci sono rating nel test set
                predicted_rating = recomm.get_user_prediction(user_id, movie_id)  # Chiamata semplificata!
                if user_id not in test_predictions_dict:
                    test_predictions_dict[user_id] = {}
                test_predictions_dict[user_id][movie_id] = predicted_rating

    mae, rmse = compute_mae_rmse(test_matrix, test_predictions_dict)

    evaluation_output.append(f"\### Risultati MAE e RMSE ###")
    evaluation_output.append(f"  MAE: {mae:.10f}")
    evaluation_output.append(f"  RMSE: {rmse:.10f}\n")

    print(f"\### Risultati MAE e RMSE ###")
    print(f"  MAE: {mae:.10f}")
    print(f"  RMSE: {rmse:.10f}\n")

    # Salva i risultati su file
    model_name = f"knn_model_NN{NN}_pearson_user_matrix"  # Nome modello più descrittivo
    with open(f"results/{model_name}.txt", "w") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
        for line in evaluation_output:
            f.write(line + "\n")
        f.write("=" * 70 + "\n")

    # Salva ogni modello (se necessario, potresti salvare solo la matrice di similarità)
    model_path = f"models/{model_name}.pkl"
    recomm.save_model(model_path)  # Dovresti adattare save_model per salvare la matrice se necessario


if __name__ == "__main__":
    import os
    import datetime
    import numpy as np

    eval_cf_user()
