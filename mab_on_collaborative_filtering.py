import logging
import numpy as np
import pandas as pd
from typing import Optional
from utils import min_max_normalize
from epsilon_mab import EpsGreedyMAB
from collaborative_filtering_recomm import CollaborativeRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _print_final_stats(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame) -> None:
    logging.info("\nFinal bandit statistics:")
    top_n_arms = bandit_mab.get_top_n()  # Returns the top N arms with their corresponding Q-values sorted
    for i, (curr_arm, q_value) in enumerate(top_n_arms):
        curr_movie_id = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title = df_recommendations.iloc[curr_arm]["title"]

        logging.info(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"with Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_total_rewards_list()[curr_arm]:.2f}"
            f" and selected {bandit_mab.get_clicks_for_arm()[curr_arm]} times"
        )


def _get_topk_movies(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame) -> None:
    logging.info("\nTop recommended movies:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, q_value) in enumerate(top_n_arms):
        curr_movie_id = df_recommendations.iloc[curr_selected_arm]["movieId"]
        topk.append(curr_movie_id)
    return topk


# *** Collaborative Filtering Item-based *** #
def compute_reward_item(similarity: float, mean_reward: float, beta: float = 0.5) -> float:
    """The reward is a linear combination of item-item similarity and the average rating of the recommended movie.
    The idea is to reward arms (movies) that are both similar to the starting movie (item-based contextual relevance)
    and, to some extent, well-rated in general (quality/popularity)."""
    return beta * similarity + (1 - beta) * mean_reward


def _start_rounds_cf_item(
    num_rounds: int,
    bandit_mab: EpsGreedyMAB,
    df_recommendations: pd.DataFrame,
    sim_scores: np.ndarray,
    df_ratings: pd.DataFrame,
) -> None:

    logging.info(f"Number of arms in bandit: {bandit_mab._n_arms}")
    logging.info(f"Number of rows in df_recommendations: {len(df_recommendations)}")
    logging.info(f"Size of sim_scores: {len(sim_scores)}")

    for i in range(0, num_rounds):

        # The bandit selects an arm
        curr_arm: int = bandit_mab.play()

        # Collaborative: Retrieve the embedding index of the movie selected by the bandit
        curr_movie_id: int = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_recommendations.iloc[curr_arm]["title"]

        # Get the similarity score for the selected movie (from sim_scores vector)
        curr_similarity: float = sim_scores[curr_arm]

        # Calculate the normalized mean for the selected movie
        movie_ratings: pd.Series = df_ratings[df_ratings["movieId"] == curr_movie_id]["rating"]
        curr_mean: float = min_max_normalize(movie_ratings, min_val=0.5, max_val=5.0)

        # Calculate the hybrid reward
        hybrid_reward = compute_reward_item(curr_similarity, curr_mean, beta=0.8)

        """
        logging.info(f"Round {i}:")
        logging.info(f"  - Selected arm: {curr_arm} -> MovieId: {curr_movie_id}, title: {curr_movie_title}")
        logging.info(f"  - Similarity: {curr_similarity:.3f}, Normalized Mean: {curr_mean:.3f}, Hybrid reward: {hybrid_reward:.3f}")
        logging.info()'
        """

        # Update the bandit with the calculated reward
        bandit_mab.update(curr_arm, hybrid_reward)


def _mab_on_collabfilter_item(
    recomm: CollaborativeRecommender,
    movie_id: int,
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    num_rounds: int = 1000,
    NN: int = 20,
) -> None:

    # Get the DataFrame of recommended movies
    df_recommendations: pd.DataFrame = recomm.get_item_recommendations(movie_id, df_movies).head(NN)
    recomm_movie_ids: pd.Index = df_recommendations.index

    # Reset the DataFrame index of recommendations to make it compatible with the bandit
    df_recommendations.reset_index(drop=False, inplace=True)

    # Retrieve similarity between movie_id and recomm_movie_ids
    sim_scores = recomm.dist_items[recomm_movie_ids].to_numpy()

    # Instantiation of Epsilon-Greedy MAB bandit
    bandit_mab = EpsGreedyMAB(n_arms=NN, epsilon=0.1, Q0=0.0)

    # Game simulation
    _start_rounds_cf_item(num_rounds, bandit_mab, df_recommendations, sim_scores, df_ratings)

    _print_final_stats(bandit_mab, df_recommendations)

    # Retrieve the top k movies recommended with the bandit
    return _get_topk_movies(bandit_mab, df_recommendations)


# *** Collaborative Filtering User-based *** #
def _start_rounds_cf_user(
    recomm: CollaborativeRecommender,
    num_rounds: int,
    bandit_mab: EpsGreedyMAB,
    df_recommendations: pd.DataFrame,
    df_ratings: pd.DataFrame,
    user_id: int,
) -> None:

    for i in range(0, num_rounds):

        # The bandit selects an arm
        curr_arm: int = bandit_mab.play()

        # Epsilon decay
        bandit_mab._log_epsilon_decay(num_round=i, decay=0.001)

        curr_movie_id: int = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_recommendations.iloc[curr_arm]["title"]
        logging.info(f"\ncurr_selected_arm: {curr_arm}")
        logging.info(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        prediction: float = recomm.get_prediction_value_clipped(user_id, curr_movie_id, NN=20)

        """ The reward is directly proportional to the rating prediction.
        The idea is to reward arms (movies) for which the user-based model predicts 
        a higher rating for the target user."""
        reward: float = min_max_normalize(prediction, min_val=0.5, max_val=5.0)

        # logging.info(f"Round {i}:")
        # logging.info(f"  - Selected arm: {curr_arm} -> MovieId: {curr_movie_id}, title: {curr_movie_title}")
        # logging.info(f"  - Mean-Centered Prediction (before normalization): {prediction:.3f}, Reward (normalized prediction): {reward:.3f}")

        # Update the bandit with the calculated reward
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

    # Get the DataFrame of recommended movies
    df_recommendations: pd.DataFrame = recomm.get_user_recommendations(user_id, matrix, df_movies).head(NN)

    # Reset the DataFrame index of recommendations to make it compatible with the bandit
    df_recommendations.reset_index(drop=False, inplace=True)
    logging.info(f"Recommendations:\n {df_recommendations}")

    # Instantiation of Epsilon-Greedy MAB bandit
    bandit_mab = EpsGreedyMAB(n_arms=NN, epsilon=0.9, Q0=0.0)

    # Game simulation
    _start_rounds_cf_user(recomm, num_rounds, bandit_mab, df_recommendations, df_ratings, user_id)

    _print_final_stats(bandit_mab, df_recommendations)

    logging.info(f"{_get_topk_movies(bandit_mab, df_recommendations)}")

    # Retrieve the top k movies recommended with the bandit
    return _get_topk_movies(bandit_mab, df_recommendations)


def mab_on_collabfilter(
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    movie_id: Optional[int] = None,
    user_id: Optional[int] = None,
    num_rounds: int = 1000,
    N: int = 20,
    recommender: CollaborativeRecommender = None,
    utility_matrix: pd.DataFrame = None,
) -> list:
    """Simulates the bandit on a Collaborative Filtering recommender (user-based or item-based)."""
    if not movie_id and not user_id:
        raise ValueError("At least movie_id or user_id must be specified")

    # Instantiate the Recommender with the KNN model
    recomm = recommender

    if movie_id and user_id:
        # Recommendations for both
        recomm_mab_item = _mab_on_collabfilter_item(recomm, movie_id, df_ratings, df_movies, num_rounds, N)
        recomm_mab_user = _mab_on_collabfilter_user(recomm, utility_matrix, user_id, df_ratings, df_movies, num_rounds, N)
        return recomm_mab_item, recomm_mab_user
    elif movie_id and not user_id:
        # Recommendations for Item-Collaborative Filtering
        return _mab_on_collabfilter_item(recomm, movie_id, df_ratings, df_movies, num_rounds, N)
    else:
        # Recommendations for User-Collaborative Filtering
        return _mab_on_collabfilter_user(recomm, utility_matrix, user_id, df_ratings, df_movies, num_rounds, N)
