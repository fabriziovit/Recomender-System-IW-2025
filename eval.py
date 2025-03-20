import datetime
import pandas as pd
import numpy as np


def eval_mae_rmse(test_matrix: pd.DataFrame, predictions_dict: dict) -> tuple:
    """Calcola MAE e RMSE per il test set, considerando solo i rating reali."""
    absolute_errors = []
    squared_errors = []
    for user_id in test_matrix.index:
        df_predictions = predictions_dict[user_id]
        # Seleziono solo i film per cui l'utente ha un rating espliciti nel test set
        user_movies = test_matrix.columns[test_matrix.loc[user_id] > 0.0]
        for movie_id in user_movies:
            real_rating = test_matrix.loc[user_id, movie_id]
            if movie_id in df_predictions.index:
                predicted_rating = df_predictions.loc[movie_id, "values"]
            else:
                raise (f"Errore: movieId {movie_id} non presente nelle predizioni per user {user_id}. Uso valore di default.")
            error = np.abs(real_rating - predicted_rating)
            squared_error = np.square(error)
            absolute_errors.append(error)
            squared_errors.append(squared_error)
    mae = np.mean(absolute_errors) if absolute_errors else 0
    rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0
    return mae, rmse


def eval_precision_recall(train_ratings_matrix, test_ratings_matrix, K_list, relevant_value, precomputed_predictions: dict) -> dict:
    """Valuta il recommender user-based utilizzando Precision@K e Recall@K per ciascun valore di K in K_list."""
    results = {}
    evaluation_output = []
    evaluation_output.append(f"Parametri valutazione: K_list={K_list},  relevant_value={relevant_value}")
    evaluation_output.append(f"test_matrix.shape: {test_ratings_matrix.shape}, train_matrix.shape: {train_ratings_matrix.shape}\n")

    # Gli utenti comuni sono già noti
    common_users = precomputed_predictions.keys()

    for K in K_list:
        precision_sum = 0
        recall_sum = 0
        user_count = 0
        for user_id in common_users:
            # Film rilevanti nel test set per l'utente (rating >= relevant_value)
            test_user_ratings = test_ratings_matrix.loc[user_id]
            relevant_items_in_test = test_user_ratings[test_user_ratings >= relevant_value].index
            if len(relevant_items_in_test) == 0:
                print(f"  Nessun film rilevante per user {user_id} nel test set. Salto.")
                continue
            # Usa le predizioni precompute per l'utente
            predictions_df = precomputed_predictions[user_id]
            # Prendi i primi K film con predizioni più alte
            recommended_items = predictions_df.head(K).index
            # Calcola True Positives come intersezione tra raccomandati e film rilevanti
            true_positives = len(set(recommended_items) & set(relevant_items_in_test))
            precision = true_positives / K
            recall = true_positives / len(relevant_items_in_test)
            evaluation_output.append(f"  User {user_id}: Precision@{K} = {precision:.10f}, Recall@{K} = {recall:.10f}")
            precision_sum += precision
            recall_sum += recall
            user_count += 1
        if user_count > 0:
            avg_precision = precision_sum / user_count
            avg_recall = recall_sum / user_count
        else:
            avg_precision, avg_recall = 0, 0

        evaluation_output.append(f"\n@@@ Test Valutazione User-Based (con K = {K} e value = {relevant_value}) @@@")
        evaluation_output.append(f"Numero utenti valutati: {user_count}")
        evaluation_output.append(f"Precision@{K}: {avg_precision:.10f}")
        evaluation_output.append(f"Recall@{K}: {avg_recall:.10f}\n")
        print(f"\n@@@ Test Valutazione User-Based (con K = {K} e value = {relevant_value}) @@@")
        print(f"Numero utenti valutati: {user_count}")
        print(f"Precision@{K}: {avg_precision:.10f}")
        print(f"Recall@{K}: {avg_recall:.10f}\n")
        results[K] = (avg_precision, avg_recall)

    # Scrittura del report (opzionale)
    with open(f"knn_acc{relevant_value}_NN{K}.txt", "w") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
        for line in evaluation_output:
            f.write(line + "\n")
        f.write("=" * 70 + "\n")
    return results
