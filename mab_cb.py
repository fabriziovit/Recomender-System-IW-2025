import logging
import numpy as np
import pandas as pd
from utils import min_max_normalize
from epsilon_mab import EpsGreedyMAB
from cb_recommender import ContentBasedRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _print_info_rounds(i: int, curr_arm: int, curr_idx: int, curr_movie_id: int, curr_movie_title: str, curr_sim: float, curr_mean: float, reward: float) -> None:
    logging.info(f"Round {i}:")
    logging.info(f"  - Braccio selezionato: {curr_arm} -> Index: {curr_idx}, MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
    logging.info(f"  - Similarità: {curr_sim:.3f}, Mean Normalizzata: {curr_mean:.3f}, Hybrid reward: {reward:.3f}")
    logging.info()


def _print_final_stats(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index) -> None:
    logging.info("\nStatistiche finali del bandit:")
    top_n_arms = bandit_mab.get_top_n()  # Restituisce i top N bracci con i relativi Q-values ordinati
    for i, (curr_arm, q_value) in enumerate(top_n_arms):
        curr_idx_embedd = indexes_of_embedd[curr_arm]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title: str = df_recommendations.loc[curr_idx_embedd]["title"]
        logging.info(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"con Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_total_rewards_list()[curr_arm]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[curr_arm]} volte"
        )


def _get_topk_movies(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index) -> None:
    logging.info("\nTop film raccomandati:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, q_value) in enumerate(top_n_arms):
        curr_idx_embedd = indexes_of_embedd[curr_selected_arm]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        topk.append(curr_movie_id)
    return topk


def compute_reward(similarity: float, mean_reward: float, beta: float = 0.5) -> float:
    """La reward è una combinazione lineare della similarità (ora content-based) e della media dei rating.
    L'obiettivo è bilanciare la pertinenza del contenuto con una misura di qualità o popolarità.
    NOTA: Se non considerassimo mean_reward, il MAB diventerebbe un semplice meccanismo di selezione
    epsilon-greedy sopra un sistema di ranking statico (content-based"""
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

        # Il bandit seleziona un braccio
        curr_selected_arm: int = bandit_mab.play()

        # Content-Based: Recupera l'indice dell'embedding del film selezionato dal bandit
        curr_idx_embedd: int = indexes_embedd_of_similiar[curr_selected_arm]
        # Recupera il movieId e il titolo del film selezionato
        curr_movie_id: int = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title: str = df_recommendations.loc[curr_idx_embedd]["title"]

        logging.info(f"\ncurr_selected_arm: {curr_selected_arm}")
        logging.info(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        # Ottieni il punteggio di similarità per il film selezionato (dal vettore sim_scores)
        curr_similarity: float = sim_scores[curr_selected_arm]

        # Calcola la media delle valutazioni per il film selezionato
        movie_ratings: pd.Series = df_ratings[df_ratings["movieId"] == curr_movie_id]["rating"]
        curr_mean_reward: float = min_max_normalize(movie_ratings, min_val=0.5, max_val=5.0)

        # Calcola la hybrid reward
        reward = compute_reward(curr_similarity, curr_mean_reward, beta=0.8)

        _print_info_rounds(i, curr_selected_arm, curr_idx_embedd, curr_movie_id, curr_movie_title, curr_similarity, curr_mean_reward, reward)

        # Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_selected_arm, reward)


def mab_on_contentbased(movie_title: str, df_ratings: pd.DataFrame, num_rounds: int = 1_000, N: int = 20, recommender: ContentBasedRecommender = None) -> list:
    """Simula su recommender content-based."""
    # Ottieni l'indice del film selezionato
    curr_movie_id = recommender.df[recommender.df["title"] == movie_title]["movieId"]
    curr_idx_embedd = recommender.get_idx(movie_title)

    # Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recommender.recommend(movie_title, N)[["movieId", "title"]]
    indexes_of_embedd: pd.Index = df_recommendations.index
    logging.info(f"Reccomendations:\n {df_recommendations}")

    # Calcola i punteggi di similarità tra il film e quelli raccomandati
    sim_scores_items: np.ndarray = recommender.compute_similarity_scores(curr_idx_embedd)
    sim_scores_items: np.ndarray = sim_scores_items[indexes_of_embedd]

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=N, epsilon=0.1, Q0=0.0)

    # Simulazione del gioco
    _start_rounds(num_rounds, bandit_mab, df_recommendations, indexes_of_embedd, sim_scores_items, df_ratings)

    _print_final_stats(bandit_mab, df_recommendations, indexes_of_embedd)

    return _get_topk_movies(bandit_mab, df_recommendations, indexes_of_embedd)
