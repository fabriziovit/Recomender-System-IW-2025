import datetime
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from done.utils import load_movielens_data, hold_out_random, pearson_distance


def precompute_user_means(users_x_movie: pd.DataFrame) -> dict:
    """
    Calcola la media dei rating per ogni utente ignorando gli 0.
    Restituisce un dizionario in cui la chiave è l'user_id e il valore è la media.
    """
    # Sostituisci gli 0 con NaN, calcola la media lungo le colonne e sostituisci eventuali NaN con 0.
    user_means = users_x_movie.replace(0, np.nan).mean(axis=1).fillna(0).to_dict()
    return user_means


def predict_rating_mean_np_migliorato(movie_id: int, user_id: int, similiars_x_movie_s: pd.Series, dist_s: pd.Series, user_means: dict) -> float:
    """
    Calcola la media pesata ignorando rating mancanti (0.0) in maniera vettoriale.
    Non sostituisce i rating 0 con le medie degli utenti.
    """
    # Converti i rating dei simili e le similarità in array NumPy
    ratings = similiars_x_movie_s.to_numpy()
    sim_uv = np.maximum(0, 1 - dist_s.to_numpy())

    # Identifica gli indici dei rating non nulli
    non_zero_rating_indices = ratings != 0

    # Applica il filtro ai rating e alle similarità
    filtered_ratings = ratings[non_zero_rating_indices]
    filtered_sim_uv = sim_uv[non_zero_rating_indices]

    # Calcola numeratore e denominatore solo per i rating non nulli
    numeratore = np.dot(filtered_sim_uv, filtered_ratings)
    denominatore = np.sum(np.abs(filtered_sim_uv))

    return numeratore / denominatore if denominatore != 0 else 0


def predict_rating_mean_np(movie_id: int, user_id: int, similiars_x_movie_s: pd.Series, dist_s: pd.Series, user_means: dict) -> float:
    """
    Calcola la media pesata NON ignorando rating mancanti (0.0) in maniera vettoriale,
    utilizzando le medie pre-calcolate per ogni utente.
    Se il rating di un utente simile è 0, viene sostituito con la sua media pre-calcolata.
    """
    # Converti i rating dei simili in un array NumPy
    ratings = similiars_x_movie_s.to_numpy()

    # Estrai le medie degli utenti simili dal dizionario
    means = np.array([user_means[u] for u in similiars_x_movie_s.index])

    # Sostituisci i rating a 0 con la media pre-calcolata dell'utente simile
    adjusted_ratings = np.where(ratings == 0, means, ratings)

    # Calcola il vettore delle similarità: max(0, 1 - distanza)
    sim_uv = np.maximum(0, 1 - dist_s.to_numpy())

    # Calcola numeratore e denominatore in modo vettoriale
    numeratore = np.dot(sim_uv, adjusted_ratings)
    denominatore = np.sum(np.abs(sim_uv))

    return numeratore / denominatore if denominatore != 0 else 0


def get_train_and_test_matrix(df_ratings: pd.DataFrame, all_movies_id: pd.Index):
    # Split dei dati in training e test
    df_train_ratings, df_test_ratings = hold_out_random(df_ratings)

    train_ratings_matrix = df_train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    test_ratings_matrix = df_test_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Train ratings-matrix: {train_ratings_matrix.shape}, Test ratings-matrix: {test_ratings_matrix.shape}")

    # Assicuriamoci che entrambe le matrici abbiano tutte le colonne
    train_ratings_matrix = train_ratings_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    test_ratings_matrix = test_ratings_matrix.reindex(columns=all_movies_id, fill_value=0.0)
    print(f"# Train ratings-matrix: {train_ratings_matrix.shape}, Test ratings-matrix: {test_ratings_matrix.shape}")

    return train_ratings_matrix, test_ratings_matrix


class CF_SVD_DimensionalityReduction_UserBased:

    def __init__(self, model_user: NearestNeighbors):
        """
        Inizializza il Recommender con un modello di machine learning (NearestNeighbors).
        """
        if not isinstance(model_user, NearestNeighbors):
            raise ValueError("model_user must be NearestNeighbors")
        self.model_user = model_user
        self._is_fitted_on_matrix_user = False

    def fit_user_model(self, matrix: pd.DataFrame, re_fit: bool = False) -> None:
        """
        Addestra il modello user-based (NearestNeighbors) su matrix se necessario o se forzato.
        """
        if not self._is_fitted_on_matrix_user or re_fit:
            self.model_user.fit(matrix)
            self._is_fitted_on_matrix_user = True
            print("*** Modello model_user addestrato su matrix. ***")
        else:
            print("*** Modello model_user già addestrato su matrix. Salto l'addestramento. ***")

    def get_predictions(self, user_id: int, factored_matrix: pd.DataFrame, train_matrix: pd.DataFrame, user_means: dict) -> pd.DataFrame:
        """
        Raccomanda film a un utente dato utilizzando il filtraggio collaborativo user-based.
        """
        all_users = train_matrix.index

        # 1. Seleziona le caratteristiche dell'utente specificato dalla matrice utenti-film
        user_index = train_matrix.index.get_loc(user_id)
        user_features = factored_matrix[user_index].reshape(1, -1)

        # 2. Trova gli utenti simili usando NearestNeighbors
        distances, pos_indexes = self.model_user.kneighbors(user_features)

        # 3. Esclude l'utente target (la prima posizione)
        similar_users = [all_users[pos_idx] for pos_idx in pos_indexes.squeeze().tolist()[1:]]
        dist = pd.Series(distances.squeeze().tolist()[1:], index=similar_users)
        print(f"#Utenti simili per user {user_id}: {similar_users} con distanze: " + f"{[str(uid) + ': ' + str(val) for uid, val in dist.items()]}")

        # 4. Recupera le valutazioni degli utenti simili
        users_x_movies_similar_df = train_matrix.loc[similar_users]

        # 5. Rimuove i film già visti dall'utente target dalla lista dei raccomandabili
        seen_mask = train_matrix.loc[user_id] > 0
        seen_movies = train_matrix.loc[user_id][seen_mask].index
        users_x_movies_similar_df = users_x_movies_similar_df.loc[:, ~users_x_movies_similar_df.columns.isin(seen_movies)]

        # 6. I film raccomandabili sono le colonne rimaste
        res_movies_ids = users_x_movies_similar_df.columns

        # 7. Per ogni film, predice il rating basandosi sui rating degli utenti simili
        #! Devo provare l'altra funzione predict_rating_mean_np_migliorata!
        values = [predict_rating_mean_np(mid, user_id, users_x_movies_similar_df[mid], dist, user_means) for mid in res_movies_ids]

        return pd.DataFrame(values, index=res_movies_ids, columns=["values"]).sort_values(by="values", ascending=False)


def precompute_all_predictions(recomm: CF_SVD_DimensionalityReduction_UserBased, train_ratings_matrix: pd.DataFrame, utility_matrix: pd.DataFrame, common_users: set, user_means: dict) -> dict:
    """
    Calcola e salva le predizioni per tutti gli utenti in common_users.
    Restituisce un dizionario: {user_id: predictions_df}
    """
    all_user_predictions = {}
    for user_id in common_users:
        all_user_predictions[user_id] = recomm.get_predictions(user_id, train_ratings_matrix, utility_matrix, user_means)
    return all_user_predictions


def evaluate_user_based(recomm: CF_SVD_DimensionalityReduction_UserBased, train_ratings_matrix, test_ratings_matrix, K_list, NN, relevant_value, n_comp, n_iter, precomputed_predictions: dict) -> dict:
    """
    Valuta il recommender user-based utilizzando Precision@K e Recall@K per ciascun valore di K in K_list.
    """
    results = {}
    evaluation_output = []
    evaluation_output.append(f"Parametri valutazione: K_list={K_list}, n_neighbors={NN}, relevant_value={relevant_value}")
    evaluation_output.append(f"test_matrix.shape: {test_ratings_matrix.shape}, train_matrix.shape: {train_ratings_matrix.shape}\n")

    # Gli utenti comuni sono già noti
    common_users = precomputed_predictions.keys()

    for K in K_list:
        precision_sum = 0
        recall_sum = 0
        user_count = 0

        evaluation_output.append(f"Valutazioni per K = {K}")
        for user_id in common_users:
            evaluation_output.append(f"Valutazione per user {user_id} con: [value = {relevant_value}, NN = {NN} e K = {K}]")
            print(f"\n### Valutazione per user {user_id} con [value = {relevant_value}, NN = {NN} e K = {K}] ###")

            # Film rilevanti nel test set per l'utente (rating >= relevant_value)
            test_user_ratings = test_ratings_matrix.loc[user_id]
            relevant_items_in_test = test_user_ratings[test_user_ratings >= relevant_value].index
            if len(relevant_items_in_test) == 0:
                evaluation_output.append(f"  Nessun film rilevante per user {user_id} nel test set. Salto.")
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
            evaluation_output.append(f"  User {user_id}: Precision@{K} = {precision:.4f}, Recall@{K} = {recall:.4f}")

            precision_sum += precision
            recall_sum += recall
            user_count += 1

        if user_count > 0:
            avg_precision = precision_sum / user_count
            avg_recall = recall_sum / user_count
        else:
            avg_precision, avg_recall = 0, 0

        evaluation_output.append(f"\n@@@ Test Valutazione User-Based (con K = {K}, NN = {NN} e value = {relevant_value}) @@@")
        evaluation_output.append(f"Numero utenti valutati: {user_count}")
        evaluation_output.append(f"Precision@{K}: {avg_precision:.4f}")
        evaluation_output.append(f"Recall@{K}: {avg_recall:.4f}\n")

        print(f"\n@@@ Test Valutazione User-Based (con K = {K}, NN = {NN} e value = {relevant_value}) @@@")
        print(f"Numero utenti valutati: {user_count}")
        print(f"Precision@{K}: {avg_precision:.4f}")
        print(f"Recall@{K}: {avg_recall:.4f}\n")

        results[K] = (avg_precision, avg_recall)

    # Scrittura del report (opzionale)
    with open(f"factorized_{relevant_value}_nn{NN}_ncomp{n_comp}_niter{n_iter}.txt", "w") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
        for line in evaluation_output:
            f.write(line + "\n")
        f.write("=" * 70 + "\n")
    return results


def calculate_mae_rmse(
    test_ratings_matrix: pd.DataFrame, train_ratings_matrix: pd.DataFrame, K_list: list, NN: int, n_comp: int, n_iter: int, precomputed_predictions: dict
) -> dict:
    """
    Calcola MAE e RMSE per il test set per ciascun valore di K in K_list.
    """
    results = {}
    evaluation_output = []
    evaluation_output.append(f"Parametri valutazione: n_neighbors={NN}")
    evaluation_output.append(f"test_matrix.shape: {test_ratings_matrix.shape}, train_matrix.shape: {train_ratings_matrix.shape}\n")

    evaluation_output.append(f"Valutazioni MAE e RMSE per K = {K_list}")
    absolute_errors = []
    squared_errors = []
    for user_id in test_ratings_matrix.index:
        evaluation_output.append(f"Valutazione per user {user_id} con NN = {NN} e per K = {K_list}")
        # Usa le predizioni precompute
        df_predictions = precomputed_predictions[user_id]

        # Film per cui l'utente ha un rating nel test
        curr_user_movies = test_ratings_matrix.columns[test_ratings_matrix.loc[user_id] > 0.0]
        for movie_id in curr_user_movies:
            real_rating = test_ratings_matrix.loc[user_id, movie_id]
            if movie_id in df_predictions.index:
                predicted_rating = df_predictions.loc[movie_id, "values"]
            else:
                print(f"  Film movieId {movie_id} non presente nelle predizioni per user {user_id}. Uso valore di default.")
                predicted_rating = train_ratings_matrix.loc[user_id].mean()

            error = np.abs(real_rating - predicted_rating)
            squared_error = np.square(error)
            absolute_errors.append(error)
            squared_errors.append(squared_error)
            evaluation_output.append(f"  User {user_id}: Errore per movieId {movie_id} = {error:.4f}, Errore^2 = {squared_error:.4f}")

    mae = np.mean(absolute_errors) if absolute_errors else 0
    rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0
    results = (mae, rmse)
    evaluation_output.append(f"\n@@@ Risultati MAE e RMSE per K = {K_list}, NN = {NN} @@@")
    evaluation_output.append(f"MAE: {mae:.4f}")
    evaluation_output.append(f"RMSE: {rmse:.4f}\n")
    print(f"\n@@@ Risultati MAE e RMSE per K = {K_list}, NN = {NN} @@@")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}\n")

    with open(f"factorized_mae_rmse_nn{NN}_ncomp{n_comp}_niter{n_iter}.txt", "w") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
        for line in evaluation_output:
            f.write(line + "\n")
        f.write("=" * 70 + "\n")
    return results


def main():
    # 1. Carica il dataset MovieLens
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")

    all_movies_id = df_movies.index

    # 2. Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Numero di utenti totali: {len(utility_matrix.index)}")
    print(f"# Numero di movies totali: {len(utility_matrix.columns)}")
    print(f"# Utility ratings-matrix: {utility_matrix.shape}")

    # 3. Precomputazione della media per ogni utente
    user_means = precompute_user_means(utility_matrix)

    # 4. Splitting in training e test matrix
    train_matrix, test_matrix = get_train_and_test_matrix(df_ratings, all_movies_id)

    best_rmse = float("inf")
    best_mae = float("inf")
    best_params = None
    results = {}
    n_components_list = [10]
    n_iter_list = [30]

    for n_components in n_components_list:
        for n_iter in n_iter_list:
            # 5. Modello SVD per la matrix-factorization
            svd = TruncatedSVD(n_components=n_components, n_iter=n_iter)

            # 6. Fit del modello SVD on Training
            factored_matrix = svd.fit_transform(train_matrix)

            # Parametri di test
            K_list = [5]
            nn_list = [5, 10]
            relevant_value = 3.0

            results_all = {}

            # Determina gli utenti comuni (che useranno le predizioni precompute)
            common_users = set(train_matrix.index) & set(test_matrix.index)
            print(f"# Numero di utenti comuni: {len(common_users)}")

            for NN in nn_list:
                knn_model = NearestNeighbors(n_neighbors=NN + 1, metric=pearson_distance, algorithm="brute", n_jobs=-1)
                recomm = CF_SVD_DimensionalityReduction_UserBased(knn_model)
                recomm.fit_user_model(factored_matrix, re_fit=True)

                # Precomputo le predizioni per tutti gli utenti comuni
                precomputed_predictions = precompute_all_predictions(recomm, factored_matrix, train_matrix, common_users, user_means)

                results_acc_rec = evaluate_user_based(recomm, train_matrix, test_matrix, K_list, NN, relevant_value, n_components, n_iter, precomputed_predictions)
                results_mae_rmse = calculate_mae_rmse(test_matrix, train_matrix, K_list, NN, n_components, n_iter, precomputed_predictions)
                results_all[NN] = {"precision_recall": results_acc_rec, "mae_rmse": results_mae_rmse}

                # Salva i risultati
                results[(n_components, n_iter)] = results_mae_rmse
                mae = results_mae_rmse[0]
                rmse = results_mae_rmse[1]

                # Se la metrica (RMSE) è migliore, aggiorna i parametri migliori
                if rmse < best_rmse and mae < best_mae:
                    best_rmse = rmse
                    best_mae = mae
                    best_params = (n_components, n_iter)

                print(f"n_components: {n_components}, n_iter: {n_iter} -> RMSE: {rmse:.4f}")

            print("Risultati complessivi:")
            print(results_all)

    print("Migliori parametri trovati:", best_params, "con RMSE:", best_rmse)


if __name__ == "__main__":
    main()
