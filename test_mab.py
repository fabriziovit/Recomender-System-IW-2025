import logging
import numpy as np
import pandas as pd
from latent_factor_model_recomm import MF_SGD_User_Based
from epsilon_mab import EpsGreedyMAB
from utils import load_movielens_data
from utils import exp_epsilon_decay, linear_epsilon_decay, log_epsilon_decay

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _print_final_stats(df_merged: pd.DataFrame, bandit_mab: EpsGreedyMAB) -> None:
    logging.info("\nFinal bandit statistics:")
    top_n_arms = bandit_mab.get_top_n()[0:20]  #! Returns the top N arms with their corresponding Q-values sorted
    for i, (curr_arm, _) in enumerate(top_n_arms):
        curr_movie_id = df_merged.iloc[curr_arm]["movieId"]
        curr_movie_title = df_merged.iloc[curr_arm]["title"]

        logging.info(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"with Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_total_rewards_list()[curr_arm]:.2f}"
            f" and selected {bandit_mab.get_clicks_for_arm()[curr_arm]} times"
        )
    logging.info(f"num_exploration: {bandit_mab._nexploration}")
    logging.info(f"num_exploitation: {bandit_mab._nexploitation}")


def _get_top_movies(bandit_mab: EpsGreedyMAB, df_merged: pd.DataFrame) -> list:
    logging.info("\nTop recommended movies:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, _) in enumerate(top_n_arms):
        curr_movie_id = df_merged.iloc[curr_selected_arm]["movieId"]
        topk.append(curr_movie_id)
    return topk


def _start_rounds(df_expected: pd.DataFrame, bandit_mab: EpsGreedyMAB, num_rounds: int, decay: bool = True) -> None:

    for i in range(0, num_rounds):

        # The bandit selects an arm (movie)
        curr_arm: int = bandit_mab.play()

        # Retrieve information about the selected movie
        curr_movie_id: int = df_expected.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_expected.iloc[curr_arm]["title"]
        logging.info(f"\ncurr_selected_arm: {curr_arm}")
        logging.info(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        # The reward corresponds to the prediction of the selected movie
        reward = df_expected.iloc[curr_arm]["values"]

        """
        logging.info(f"Round {i}:")
        logging.info(f"  - Selected arm: {curr_arm} -> MovieId: {curr_movie_id}, title: {curr_movie_title}")
        logging.info(f"  - Reward: {reward:.3f}, epsilon: {bandit_mab.get_curr_epsilon():.3f}")
        logging.info(f"  - function: {bandit_mab._epsilon_decay_function.__name__}")
        """

        if decay:
            # Update epsilon at each round
            bandit_mab.update_epsilon(i)

        # Update the bandit with the calculated reward
        bandit_mab.update(curr_arm, reward)


def mab(df_expected: pd.DataFrame, bandit_mab: EpsGreedyMAB, num_rounds: int = 10_000, decay: bool = True) -> list:

    # Game simulation
    _start_rounds(df_expected, bandit_mab, num_rounds, decay)
    _print_final_stats(df_expected, bandit_mab)

    # Retrieve the top k movies recommended with the bandit
    return _get_top_movies(bandit_mab, df_expected)


def test_single_user(
    user_id: int,
    recomm: MF_SGD_User_Based,
    utility_matrix: pd.DataFrame,
    fun: callable = None,
    epsilon: float = 0.99,
) -> None:

    # 1. Get predictions for the specified user user_id
    df_expected = recomm.get_recommendations(utility_matrix, user_id)
    df_expected = df_expected.merge(df_movies, on="movieId")[["title", "values"]].sort_index(ascending=True)
    # 2. Set the index to make it compatible with the bandit
    new_index = pd.RangeIndex(df_expected.shape[0], name="idxarm")
    df_expected.reset_index(drop=False, inplace=True)
    df_expected.set_index(new_index, inplace=True)
    # 3. Define the bandit and epsilon decay function

    bandit_mab = EpsGreedyMAB(n_arms=df_expected.shape[0], epsilon=epsilon, Q0=0.0)

    if fun is not None:
        bandit_mab.set_epsilon_deacy(fun)

    # Get recommendations
    top_k = mab(df_expected, bandit_mab, num_rounds=10_000)[:10]  #! Returns the first 10 recommended movies

    # Save results in a list
    res_user: tuple = (bandit_mab._nexploration, bandit_mab._nexploitation, bandit_mab._total_rewards.sum())
    logging.info(f"Results for user {user_id}: {res_user}")

    return res_user


def test_dacys_all_users(df_movies: pd.DataFrame, df_ratings: pd.DataFrame, df_tags: pd.DataFrame) -> None:
    # Create the utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Load the MF_SGD_User_Based model
    model_path = "./models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl"
    recomm: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path)

    for fun in [linear_epsilon_decay]:
        logging.info(f"Epsilon decay function: {fun.__name__}")

        dictionary = {"exploration": 0, "exploitation": 0, "comulative_reward": 0.0}
        for user_id in utility_matrix.index:
            exp, exploit, reward = test_single_user(user_id, recomm, utility_matrix, fun)
            dictionary["exploration"] += exp
            dictionary["exploitation"] += exploit
            dictionary["comulative_reward"] += reward

        evaluation_output = [
            f"Epsilon decay function: {fun.__name__}",
            f"Number of explorations: {dictionary['exploration']}",
            f"Number of exploitations: {dictionary['exploitation']}",
            f"Total reward: {dictionary['comulative_reward']}",
        ]
        with open(f"results/{fun.__name__}_mab.txt", "w") as f:
            f.write("\n" + "=" * 70 + "\n")
            for line in evaluation_output:
                f.write(line + "\n")
            f.write("=" * 70 + "\n")


def test_all_users(df_movies, df_ratings, df_tags):
    # Create the utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Load the MF_SGD_User_Based model
    model_path = "./models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl"
    recomm: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path)

    epsilon = 1.0  # Initial epsilon
    dictionary = {"exploration": 0, "exploitation": 0, "comulative_reward": 0.0}
    for user_id in utility_matrix.index:
        exp, exploit, reward = test_single_user(user_id, recomm, utility_matrix, None, epsilon)
        dictionary["exploration"] += exp
        dictionary["exploitation"] += exploit
        dictionary["comulative_reward"] += reward

    evaluation_output = [
        f"Epsilon Mab: {epsilon}",
        f"Number of explorations: {dictionary['exploration']}",
        f"Number of exploitations: {dictionary['exploitation']}",
        f"Total reward: {dictionary['comulative_reward']}",
    ]
    with open(f"results/{epsilon}_mab.txt", "w") as f:
        f.write("\n" + "=" * 70 + "\n")
        for line in evaluation_output:
            f.write(line + "\n")
        f.write("=" * 70 + "\n")


if __name__ == "__main__":
    # Load the MovieLens dataset
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")

    # test_dacys_all_users(df_movies, df_ratings, df_tags)

    test_all_users(df_movies, df_ratings, df_tags)
