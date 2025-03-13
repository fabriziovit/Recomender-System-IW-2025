import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from cb_recommender import ContentBasedRecommender
from cf_recommender import CollaborativeRecommender
from eps_mab import EpsGreedyMAB
from utils import load_movielens_data, pearson_distance, compute_mean_form_movie, compute_hybrid_reward


def start_rounds(
    num_rounds: int, bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_embedd_of_similiar: pd.Index, sim_scores: np.ndarray, df_ratings: pd.DataFrame
) -> None:

    for i in range(0, num_rounds):

        # 0. Il bandit seleziona un braccio
        curr_selected_arm: int = bandit_mab.play()
        # Recupera l'indice dell'embedding del film selezionato dal bandit
        curr_idx_embedd: int = indexes_embedd_of_similiar[curr_selected_arm]
        # Recupera il movieId e il titolo del film selezionato
        curr_movie_id: int = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title: str = df_recommendations.loc[curr_idx_embedd]["title"]

        # 1. Ottieni il punteggio di similarità per il film selezionato (dal vettore sim_scores)
        curr_similarity: float = sim_scores[curr_selected_arm]

        # 2. Calcola la reward media normalizzata per il film selezionato
        curr_mean_reward: float = compute_mean_form_movie(curr_movie_id, df_ratings)

        # 3. Calcola la hybrid reward
        reward = compute_hybrid_reward(curr_similarity, curr_mean_reward, beta=0.8)

        print(f"Round {i}:")
        print(f"  - Braccio selezionato: {curr_selected_arm} -> Index: {curr_idx_embedd}, MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
        print(f"  - Similarità: {curr_similarity:.3f}, Mean Normalizzata: {curr_mean_reward:.3f}, Hybrid reward: {reward:.3f}")

        # 4. Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_selected_arm, reward)


def print_stats_content(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index) -> None:
    print("\nStatistiche finali del bandit:")
    print(f"index_of_embedd: {indexes_of_embedd}")
    for arm_idx in range(0, bandit_mab.get_narms()):
        curr_idx_embedd = indexes_of_embedd[arm_idx]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title = df_recommendations.loc[curr_idx_embedd]["title"]
        print(
            f"  - Arm {arm_idx}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"con Q = {bandit_mab.get_qvalues()[arm_idx]:.2f}, reward_tot = {bandit_mab.get_rewards_list()[arm_idx]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[arm_idx]} volte"
        )


def print_topk_movies_contet(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index, k: int = 5) -> None:
    print("\nTop film raccomandati:")
    top_n_list = bandit_mab.get_top_n()
    for i, (curr_selected_arm, q_value) in enumerate(top_n_list):
        curr_idx_embedd = indexes_of_embedd[curr_selected_arm]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title = df_recommendations.loc[curr_idx_embedd]["title"]
        print(
            f"  - Top {i+1}: Arm {curr_selected_arm} (Movie ID {curr_movie_id}, '{curr_movie_title}'): "
            f"con Q = {q_value:.2f}, reward_tot = {bandit_mab.get_rewards_list()[curr_selected_arm]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[curr_selected_arm]} volte"
        )


def get_topk_movies_content(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index) -> None:
    print("\nTop film raccomandati:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, q_value) in enumerate(top_n_arms):
        curr_idx_embedd = indexes_of_embedd[curr_selected_arm]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        topk.append(curr_movie_id)
    return topk


def simulate_on_cb_recomm(movie_title: str, df_ratings: pd.DataFrame, N: int) -> None:

    # 0. Carica il dataset per il content-based recommender
    df = pd.read_csv("dataset/movies_with_abstracts_complete.csv", on_bad_lines="warn")
    recommender = ContentBasedRecommender(df, abstract_col="dbpedia_abstract", title_col="title", genres_col="genres")

    # 1. Ottieni l'indice del film selezionato
    curr_movie_id = recommender.df[recommender.df["title"] == movie_title]["movieId"]
    curr_idx_embedd = recommender.get_idx(movie_title)
    # print(f"\nidx_embedding del film corrente: {curr_idx_embedd} (con movie_id: {curr_movie_id}, titolo: {movie_title})")

    # 2. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recommender.recommend(movie_title, N)[["movieId", "title"]]
    indexes_of_embedd: pd.Index = df_recommendations.index
    # print(f"Film Raccomandati:\n{df_recommendations}\n e indexes: {indexes_of_embedd}\n")

    # 3. Calcola i punteggi di similarità tra il film e quelli raccomandati
    sim_scores: np.ndarray = recommender.compute_similarity_scores(curr_idx_embedd)
    sim_scores: np.ndarray = sim_scores[indexes_of_embedd]

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=N, epsilon=0.1, Q0=0.0)

    # Simulazione del gioco
    num_rounds = 10000
    start_rounds(num_rounds, bandit_mab, df_recommendations, indexes_of_embedd, sim_scores, df_ratings)

    # Stampa le statistiche finali del bandit
    # print_stats_content(bandit_mab, df_recommendations, indexes_of_embedd)

    # Stampa le informazioni sui top k film raccomandati
    # print_topk_movies_contet(bandit_mab, df_recommendations, indexes_of_embedd)

    return get_topk_movies_content(bandit_mab, df_recommendations, indexes_of_embedd)


def simulate_on_cf_recomm(movie_id: int, df_ratings: pd.DataFrame, df_movies: pd.DataFrame, NN=10, K=5) -> None:

    # Crea la matrice utenti-film pivot (userId x movieId)
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Numero di utenti totali: {len(utility_matrix.index)}")
    print(f"# Numero di movies totali: {len(utility_matrix.columns)}")
    print(f"# Utility ratings-matrix: {utility_matrix.shape}")

    # Inizializza il modello NearestNeighbors con metrica di correlazione di Pearson
    knn_model_pearson_item = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)
    knn_model_pearson_user = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=NN + 1, n_jobs=-1)

    # Istanzia il Recommender con il modello KNN
    recomm = CollaborativeRecommender(knn_model_pearson_item, knn_model_pearson_user)
    recomm.fit_user_model(utility_matrix, re_fit=True)  # Addestra il modello user-based
    recomm.fit_item_model(utility_matrix, re_fit=True)  # Addestra il modello item-based

    # 2. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recomm.get_item_recommendations(movie_id, df_movies)
    recomm_movie_ids: pd.Index = df_recommendations.index
    print(f"\nFilm Raccomandati:\n{df_recommendations}")

    # 3. Recupero la similarità tra movie_id e recomm_movie_ids
    sim_scores = recomm._dist_item
    sim_scores = sim_scores[recomm_movie_ids].to_numpy()

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=K, epsilon=0.1, Q0=0.0)

    # Simulazione del gioco
    num_round = 100
    start_rounds(num_round, bandit_mab, df_recommendations, recomm_movie_ids, sim_scores, df_ratings)

    # Stampa le statistiche finali del bandit
    print_stats_content(bandit_mab, df_recommendations, recomm_movie_ids)

    # Recupera i top k film raccomandati con il bandit
    topk_list = get_topk_movies_content(bandit_mab, df_recommendations, recomm_movie_ids, k=5)


def main():
    # 0. Carica i dati di MovieLens (df_movies, df_ratings, df_tags)
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")

    #! Movie to test
    temp_movie_title = "Toy Story 2 (1999)"
    # temp_movie_id = df_movies[df_movies["title"] == temp_movie_title].index[0]

    # MAB on content-based
    simulate_on_cb_recomm(temp_movie_title, df_ratings, N=20)

    # MAB on item-based collaborative filtering
    # simulate_on_cf_recomm(temp_movie_id, df_ratings, df_movies, NN=10, K=10)


if __name__ == "__main__":
    main()
