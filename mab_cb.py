import numpy as np
import pandas as pd
from cb_recommender import ContentBasedRecommender
from eps_mab import EpsGreedyMAB
from utils import load_movielens_data, pearson_distance, min_max_normalize_mean


def _print_info_rounds(i: int, curr_arm: int, curr_idx: int, curr_movie_id: int, curr_movie_title: str, curr_sim: float, curr_mean: float, reward: float) -> None:
    print(f"Round {i}:")
    print(f"  - Braccio selezionato: {curr_arm} -> Index: {curr_idx}, MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
    print(f"  - Similarità: {curr_sim:.3f}, Mean Normalizzata: {curr_mean:.3f}, Hybrid reward: {reward:.3f}")
    print()


def _print_final_stats(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index) -> None:
    print("\nStatistiche finali del bandit:")
    top_n_arms = bandit_mab.get_top_n()  # Restituisce i top N bracci con i relativi Q-values ordinati
    for i, (curr_arm, q_value) in enumerate(top_n_arms):
        curr_idx_embedd = indexes_of_embedd[curr_arm]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title: str = df_recommendations.loc[curr_idx_embedd]["title"]
        print(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"con Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_rewards_list()[curr_arm]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[curr_arm]} volte"
        )


def _get_topk_movies(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index) -> None:
    print("\nTop film raccomandati:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, q_value) in enumerate(top_n_arms):
        curr_idx_embedd = indexes_of_embedd[curr_selected_arm]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        topk.append(curr_movie_id)
    return topk


def compute_hybrid_reward_content(similarity: float, mean_reward: float, beta: float = 0.5) -> float:
    """Calcola della reward: combinazione lineare di similarità e rating medio del film selezionato"""
    return beta * similarity + (1 - beta) * mean_reward


def _start_rounds(
    num_rounds: int,
    bandit_mab: EpsGreedyMAB,
    df_recommendations: pd.DataFrame,
    indexes_embedd_of_similiar: pd.Index,
    sim_scores: np.ndarray,
    df_ratings: pd.DataFrame,
) -> None:

    for i in range(0, num_rounds):

        # 0. Il bandit seleziona un braccio
        curr_selected_arm: int = bandit_mab.play()

        # Content-Based: Recupera l'indice dell'embedding del film selezionato dal bandit
        curr_idx_embedd: int = indexes_embedd_of_similiar[curr_selected_arm]
        # Recupera il movieId e il titolo del film selezionato
        curr_movie_id: int = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title: str = df_recommendations.loc[curr_idx_embedd]["title"]

        print(f"\ncurr_selected_arm: {curr_selected_arm}")
        print(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        # 1. Ottieni il punteggio di similarità per il film selezionato (dal vettore sim_scores)
        curr_similarity: float = sim_scores[curr_selected_arm]

        # 2. Calcola la reward media normalizzata per il film selezionato
        curr_mean_reward: float = min_max_normalize_mean(curr_movie_id, df_ratings)

        # 3. Calcola la hybrid reward
        reward = compute_hybrid_reward_content(curr_similarity, curr_mean_reward, beta=0.8)

        _print_info_rounds(i, curr_selected_arm, curr_idx_embedd, curr_movie_id, curr_movie_title, curr_similarity, curr_mean_reward, reward)

        # 4. Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_selected_arm, reward)


def mab_on_contentbased(movie_title: str, df_ratings: pd.DataFrame, num_round: int, N: int) -> list:

    # 0. Carica il dataset per il content-based recommender
    df = pd.read_csv("dataset/movies_with_abstracts_complete.csv", on_bad_lines="warn")
    recommender = ContentBasedRecommender(df, abstract_col="dbpedia_abstract", title_col="title", genres_col="genres")

    # 1. Ottieni l'indice del film selezionato
    curr_movie_id = recommender.df[recommender.df["title"] == movie_title]["movieId"]
    curr_idx_embedd = recommender.get_idx(movie_title)

    # 2. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recommender.recommend(movie_title, N)[["movieId", "title"]]
    indexes_of_embedd: pd.Index = df_recommendations.index
    print(f"Reccomendations:\n {df_recommendations}")

    # 3. Calcola i punteggi di similarità tra il film e quelli raccomandati
    sim_scores_items: np.ndarray = recommender.compute_similarity_scores(curr_idx_embedd)
    sim_scores_items: np.ndarray = sim_scores_items[indexes_of_embedd]

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=N, epsilon=0.1, Q0=0.0)

    # Simulazione del gioco
    _start_rounds(num_round, bandit_mab, df_recommendations, indexes_of_embedd, sim_scores_items, df_ratings)

    _print_final_stats(bandit_mab, df_recommendations, indexes_of_embedd)

    return _get_topk_movies(bandit_mab, df_recommendations, indexes_of_embedd)


def main():
    # 0. Carica i dati di MovieLens (df_movies, df_ratings, df_tags)
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")
    print(f"Dataset caricato: {len(df_movies)} film, {len(df_ratings)} voti, {len(df_tags)} tag")

    # Movie to test
    temp_movie_title = "Toy Story 2 (1999)"
    temp_movie_id = df_movies[df_movies["title"] == temp_movie_title].index[0]

    # MAB on content-based
    print(mab_on_contentbased(temp_movie_title, df_ratings, num_round=1_000, N=20))


if __name__ == "__main__":
    main()
