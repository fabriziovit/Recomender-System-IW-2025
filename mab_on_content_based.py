import logging
import numpy as np
import pandas as pd
from utils import min_max_normalize
from epsilon_mab import EpsGreedyMAB
from content_based_recomm import ContentBasedRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _print_info_rounds(i: int, curr_arm: int, curr_idx: int, curr_movie_id: int, curr_movie_title: str, curr_sim: float, curr_mean: float, reward: float) -> None:
    logging.info(f"Round {i}:")
    logging.info(f"  - Selected arm: {curr_arm} -> Index: {curr_idx}, MovieId: {curr_movie_id}, title: {curr_movie_title}")
    logging.info(f"  - Similarity: {curr_sim:.3f}, Normalized Mean: {curr_mean:.3f}, Hybrid reward: {reward:.3f}")


def _print_final_stats(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index) -> None:
    logging.info("\nFinal bandit statistics:")
    top_n_arms = bandit_mab.get_top_n()  # Returns the top N arms with their corresponding Q-values sorted
    for i, (curr_arm, q_value) in enumerate(top_n_arms):
        curr_idx_embedd = indexes_of_embedd[curr_arm]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title: str = df_recommendations.loc[curr_idx_embedd]["title"]
        logging.info(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"with Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_total_rewards_list()[curr_arm]:.2f}"
            f" and selected {bandit_mab.get_clicks_for_arm()[curr_arm]} times"
        )


def _get_topk_movies(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index) -> list:
    logging.info("\nTop recommended movies:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, q_value) in enumerate(top_n_arms):
        curr_idx_embedd = indexes_of_embedd[curr_selected_arm]
        curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        topk.append(curr_movie_id)
    return topk


def compute_reward(similarity: float, mean_reward: float, beta: float = 0.5) -> float:
    """The reward is a linear combination of similarity (now content-based) and the average rating.
    The goal is to balance content relevance with a measure of quality or popularity.
    NOTE: If we didn't consider mean_reward, the MAB would become a simple epsilon-greedy
    selection mechanism over a static ranking system (content-based)."""
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

        # The bandit selects an arm
        curr_selected_arm: int = bandit_mab.play()

        # Content-Based: Retrieve the embedding index of the movie selected by the bandit
        curr_idx_embedd: int = indexes_embedd_of_similiar[curr_selected_arm]
        # Retrieve the movieId and title of the selected movie
        curr_movie_id: int = df_recommendations.loc[curr_idx_embedd]["movieId"]
        curr_movie_title: str = df_recommendations.loc[curr_idx_embedd]["title"]

        logging.info(f"\ncurr_selected_arm: {curr_selected_arm}")
        logging.info(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        # Get the similarity score for the selected movie (from sim_scores vector)
        curr_similarity: float = sim_scores[curr_selected_arm]

        # Calculate the average rating for the selected movie
        movie_ratings: pd.Series = df_ratings[df_ratings["movieId"] == curr_movie_id]["rating"]
        curr_mean_reward: float = min_max_normalize(movie_ratings, min_val=0.5, max_val=5.0)

        # Calculate the hybrid reward
        reward = compute_reward(curr_similarity, curr_mean_reward, beta=0.8)

        _print_info_rounds(i, curr_selected_arm, curr_idx_embedd, curr_movie_id, curr_movie_title, curr_similarity, curr_mean_reward, reward)

        # Update the bandit with the calculated reward
        bandit_mab.update(curr_selected_arm, reward)


def mab_on_contentbased(movie_title: str, df_ratings: pd.DataFrame, num_rounds: int = 1_000, N: int = 20, recommender: ContentBasedRecommender = None) -> list:
    """Simulates on content-based recommender."""
    # Get the index of the selected movie
    curr_movie_id = recommender.df[recommender.df["title"] == movie_title]["movieId"]
    curr_idx_embedd = recommender.get_idx(movie_title)

    # Get the DataFrame of recommended movies
    df_recommendations: pd.DataFrame = recommender.recommend(movie_title, N)[["movieId", "title"]]
    indexes_of_embedd: pd.Index = df_recommendations.index
    logging.info(f"Recommendations:\n {df_recommendations}")

    # Calculate similarity scores between the movie and those recommended
    sim_scores_items: np.ndarray = recommender.compute_similarity_scores(curr_idx_embedd)
    sim_scores_items: np.ndarray = sim_scores_items[indexes_of_embedd]

    # Instantiation of Epsilon-Greedy MAB bandit
    bandit_mab = EpsGreedyMAB(n_arms=N, epsilon=0.1, Q0=0.0)

    # Game simulation
    _start_rounds(num_rounds, bandit_mab, df_recommendations, indexes_of_embedd, sim_scores_items, df_ratings)

    _print_final_stats(bandit_mab, df_recommendations, indexes_of_embedd)

    return _get_topk_movies(bandit_mab, df_recommendations, indexes_of_embedd)
