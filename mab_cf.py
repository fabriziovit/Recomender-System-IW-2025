from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from cf_recommender import CollaborativeRecommender
from eps_mab import EpsGreedyMAB
from utils import load_movielens_data, pearson_distance, min_max_normalize


def _print_final_stats(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame) -> None:
    print("\nStatistiche finali del bandit:")
    top_n_arms = bandit_mab.get_top_n()  # Restituisce i top N bracci con i relativi Q-values ordinati
    for i, (curr_arm, q_value) in enumerate(top_n_arms):
        curr_movie_id = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title = df_recommendations.iloc[curr_arm]["title"]

        print(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"con Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_rewards_list()[curr_arm]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[curr_arm]} volte"
        )


def _get_topk_movies(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame) -> None:
    print("\nTop film raccomandati:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, q_value) in enumerate(top_n_arms):
        curr_movie_id = df_recommendations.iloc[curr_selected_arm]["movieId"]
        topk.append(curr_movie_id)
    return topk


"""Collaborative Filtering Item-based **************************************************************************************************"""


def compute_reward_item(similarity: float, mean_reward: float, beta: float = 0.5) -> float:
    """Calcola della reward: combinazione lineare di similarità e rating medio del film selezionato"""
    return beta * similarity + (1 - beta) * mean_reward


def _start_rounds_cf_item(
    num_rounds: int,
    bandit_mab: EpsGreedyMAB,
    df_recommendations: pd.DataFrame,
    sim_scores: np.ndarray,
    df_ratings: pd.DataFrame,
) -> None:

    for i in range(0, num_rounds):

        # 0. Il bandit seleziona un braccio
        curr_arm: int = bandit_mab.play()

        # Collaborative: Recupera l'indice dell'embedding del film selezionato dal bandit
        curr_movie_id: int = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_recommendations.iloc[curr_arm]["title"]

        print(f"\ncurr_selected_arm: {curr_arm}")
        print(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        # 1. Ottieni il punteggio di similarità per il film selezionato (dal vettore sim_scores)
        curr_similarity: float = sim_scores[curr_arm]

        # 2. Calcola la media normalizzata per il film selezionato
        movie_ratings: pd.Series = df_ratings[df_ratings["movieId"] == curr_movie_id]["rating"]
        curr_mean: float = min_max_normalize(movie_ratings, min_val=0.5, max_val=5.0)

        # 3. Calcola la hybrid reward
        hybrid_reward = compute_reward_item(curr_similarity, curr_mean, beta=0.8)

        print(f"Round {i}:")
        print(f"  - Braccio selezionato: {curr_arm} -> MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
        print(f"  - Similarità: {curr_similarity:.3f}, Mean Normalizzata: {curr_mean:.3f}, Hybrid reward: {hybrid_reward:.3f}")
        print()

        # 4. Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_arm, hybrid_reward)


def _mab_on_collabfilter_item(
    recomm: CollaborativeRecommender,
    movie_id: int,
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    num_rounds: int = 1000,
    NN: int = 20,
) -> None:

    # 1. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recomm.get_item_recommendations(movie_id, df_movies).head(NN)
    recomm_movie_ids: pd.Index = df_recommendations.index

    # 2. Resetta l'indice del DataFrame delle raccomandazioni per renderlo compatibile con il bandit
    df_recommendations.reset_index(drop=False, inplace=True)
    print(f"df_recommendations:\n {df_recommendations}")

    # 3. Recupero la similarità tra movie_id e recomm_movie_ids
    sim_scores = recomm._sim_items
    sim_scores = sim_scores[recomm_movie_ids].to_numpy()

    # 4. Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=NN, epsilon=0.1, Q0=0.0)

    # Simulazione del gioco
    _start_rounds_cf_item(num_rounds, bandit_mab, df_recommendations, sim_scores, df_ratings)

    # Recupera i top k film raccomandati con il bandit
    return _get_topk_movies(bandit_mab, df_recommendations)


"""Collaborative Filtering User-based **************************************************************************************************"""


def _start_rounds_cf_user(
    recomm: CollaborativeRecommender,
    num_rounds: int,
    bandit_mab: EpsGreedyMAB,
    df_recommendations: pd.DataFrame,
    df_ratings: pd.DataFrame,
    user_id: int,
) -> None:

    for i in range(0, num_rounds):

        # 0. Il bandit seleziona un braccio
        curr_arm: int = bandit_mab.play()

        # Epsilon decay
        bandit_mab.linear_epsilon_decay(num_round=i, decay=0.001)

        curr_movie_id: int = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_recommendations.iloc[curr_arm]["title"]
        print(f"\ncurr_selected_arm: {curr_arm}")
        print(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        prediction: float = recomm.get_user_predictions(df_ratings, recomm._sim_users, user_id, curr_movie_id)

        # Normalizzazione della predizione (opzionale)
        normalized_prediction: float = min_max_normalize(prediction, min_val=0.5, max_val=5.0)

        # Calcola la reward
        reward = normalized_prediction

        print(f"Round {i}:")
        print(f"  - Braccio selezionato: {curr_arm} -> MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
        print(f"  - Mean-Centered Prediction (before normalization): {prediction:.3f}, Reward (normalized prediction): {reward:.3f}")

        # 4. Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_arm, reward)


def _mab_on_collabfilter_user(
    recomm: CollaborativeRecommender,
    matrix: pd.DataFrame,
    user_id: int,
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    num_rounds: int = 1000,
    NN: int = 20,
) -> None:

    # 1. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recomm.get_user_recommendations(user_id, matrix, df_movies).head(NN)

    # 2. Resetta l'indice del DataFrame delle raccomandazioni per renderlo compatibile con il bandit
    df_recommendations.reset_index(drop=False, inplace=True)
    print(f"Reccomendations:\n {df_recommendations}")

    # 4. Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=NN, epsilon=0.9, Q0=0.0)

    # Simulazione del gioco
    _start_rounds_cf_user(recomm, num_rounds, bandit_mab, df_recommendations, df_ratings, user_id)

    _print_final_stats(bandit_mab, df_recommendations)

    print(f"Top k film raccomandati: {_get_topk_movies(bandit_mab, df_recommendations)}")

    # Recupera i top k film raccomandati con il bandit
    return _get_topk_movies(bandit_mab, df_recommendations)


def mab_on_collabfilter(
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    movie_id: Optional[int] = None,
    user_id: Optional[int] = None,
    num_rounds: int = 1000,
    N: int = 20,
) -> list:

    if not movie_id and not user_id:
        raise ValueError("Almeno movie_id o user_id devono essere specificati")

    # Crea la matrice utenti-film pivot (userId x movieId)
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Numero di utenti totali: {len(utility_matrix.index)}")
    print(f"# Numero di movies totali: {len(utility_matrix.columns)}")
    print(f"# Utility-matrix.shape: {utility_matrix.shape}")

    # Inizializza il modello NearestNeighbors con metrica di correlazione di Pearson
    knn_model_pearson_item = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=N + 1, n_jobs=-1)
    knn_model_pearson_user = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=N + 1, n_jobs=-1)

    # Istanzia il Recommender con il modello KNN
    recomm = CollaborativeRecommender(knn_model_pearson_item, knn_model_pearson_user)
    recomm.fit_user_model(utility_matrix, re_fit=True)  # Addestra il modello user-based
    recomm.fit_item_model(utility_matrix, re_fit=True)  # Addestra il modello item-based

    if movie_id and user_id:
        # Raccomandazioni per entrambi
        recomm_mab_item = _mab_on_collabfilter_item(recomm, movie_id, df_ratings, df_movies, num_rounds, N)
        recomm_mab_user = _mab_on_collabfilter_user(recomm, utility_matrix, user_id, df_ratings, df_movies, num_rounds, N)
        return recomm_mab_item, recomm_mab_user
    elif movie_id and not user_id:
        # Raccomandazioni per Item-Colaborative Filtering
        return _mab_on_collabfilter_item(recomm, movie_id, df_ratings, df_movies, num_rounds, N)
    else:
        # Raccomandazioni per User-Colaborative Filtering
        return _mab_on_collabfilter_user(recomm, utility_matrix, user_id, df_ratings, df_movies, num_rounds, N)


def main():
    # 0. Carica i dati di MovieLens (df_movies, df_ratings, df_tags)
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")
    print(f"Dataset caricato: {len(df_movies)} film, {len(df_ratings)} voti, {len(df_tags)} tag")

    # User to test
    temp_user_id = 1
    # Movie to test
    temp_movie_title = "Toy Story 2 (1999)"
    temp_movie_id = df_movies[df_movies["title"] == temp_movie_title].index[0]

    # MAB on collaborative-filtering:
    # Controllare perché restituisce sempre i risultati nello stesso ordine!
    # Probabile che ci sia un problema con il calcolo della hybrid reward

    # Item-based collaborative filtering
    # print(mab_on_collabfilter(df_ratings, df_movies, temp_movie_id, None, N=20))

    # User-based collaborative filtering
    print(mab_on_collabfilter(df_ratings, df_movies, None, temp_user_id, num_rounds=1000, N=20))


if __name__ == "__main__":
    main()
