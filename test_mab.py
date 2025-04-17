import logging
import numpy as np
import pandas as pd
from mf_sgd import MF_SGD_User_Based
from epsilon_mab import EpsGreedyMAB
from utils import load_movielens_data
from utils import exp_epsilon_decay, linear_epsilon_decay, log_epsilon_decay

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _print_final_stats(df_merged: pd.DataFrame, bandit_mab: EpsGreedyMAB) -> None:
    logging.info("\nStatistiche finali del bandit:")
    top_n_arms = bandit_mab.get_top_n()[0:20]  #! Restituisce i top N bracci con i relativi Q-values ordinati
    for i, (curr_arm, _) in enumerate(top_n_arms):
        curr_movie_id = df_merged.iloc[curr_arm]["movieId"]
        curr_movie_title = df_merged.iloc[curr_arm]["title"]

        logging.info(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"con Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_total_rewards_list()[curr_arm]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[curr_arm]} volte"
        )
    logging.info(f"num_exploration: {bandit_mab._nexploration}")
    logging.info(f"num_exploitation: {bandit_mab._nexploitation}")


def _get_top_movies(bandit_mab: EpsGreedyMAB, df_merged: pd.DataFrame) -> list:
    logging.info("\nTop film raccomandati:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, _) in enumerate(top_n_arms):
        curr_movie_id = df_merged.iloc[curr_selected_arm]["movieId"]
        topk.append(curr_movie_id)
    return topk


def _start_rounds(df_expected: pd.DataFrame, bandit_mab: EpsGreedyMAB, num_rounds: int, decay: bool = True) -> None:

    for i in range(0, num_rounds):

        # Il bandit seleziona un braccio (film)
        curr_arm: int = bandit_mab.play()

        # Recupero informazioni sul film selezionato
        curr_movie_id: int = df_expected.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_expected.iloc[curr_arm]["title"]
        logging.info(f"\ncurr_selected_arm: {curr_arm}")
        logging.info(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        # La reward corrisponde alla predizione del film selezionato
        reward = df_expected.iloc[curr_arm]["values"]

        """
        logging.info(f"Round {i}:")
        logging.info(f"  - Braccio selezionato: {curr_arm} -> MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
        logging.info(f"  - Reward: {reward:.3f}, epsilon: {bandit_mab.get_curr_epsilon():.3f}")
        logging.info(f"  - function: {bandit_mab._epsilon_decay_function.__name__}")
        """

        if decay:
            # Aggiorna epsilon ad ogni round
            bandit_mab.update_epsilon(i)

        # Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_arm, reward)


def mab(df_expected: pd.DataFrame, bandit_mab: EpsGreedyMAB, num_rounds: int = 10_000, decay: bool = True) -> list:

    # Simulazione del gioco
    _start_rounds(df_expected, bandit_mab, num_rounds, decay)
    _print_final_stats(df_expected, bandit_mab)

    # Recupera i top k film raccomandati con il bandit
    return _get_top_movies(bandit_mab, df_expected)


def test_single_user(
    user_id: int,
    recomm: MF_SGD_User_Based,
    utility_matrix: pd.DataFrame,
    fun: callable = None,
    epsilon: float = 0.99,
) -> None:

    # 1. Ottieni le predizioni per l'utente specificato user_id
    df_expected = recomm.get_recommendations(utility_matrix, user_id)
    df_expected = df_expected.merge(df_movies, on="movieId")[["title", "values"]].sort_index(ascending=True)
    # 2. Imposta l'indice per renderlo compatibile con il bandit
    new_index = pd.RangeIndex(df_expected.shape[0], name="idxarm")
    df_expected.reset_index(drop=False, inplace=True)
    df_expected.set_index(new_index, inplace=True)
    # 3. Definisci il bandit e l'epsilon decay function

    bandit_mab = EpsGreedyMAB(n_arms=df_expected.shape[0], epsilon=epsilon, Q0=0.0)

    if fun is not None:
        bandit_mab.set_epsilon_deacy(fun)

    # Ottieni le raccomandazioni
    top_k = mab(df_expected, bandit_mab, num_rounds=10_000)[:10]  #! Restituisce i primi 10 film raccomandati

    # Salvo i risultati in una lista
    res_user: tuple = (bandit_mab._nexploration, bandit_mab._nexploitation, bandit_mab._total_rewards.sum())
    logging.info(f"Results for user {user_id}: {res_user}")

    return res_user


def test_dacys_all_users(df_movies: pd.DataFrame, df_ratings: pd.DataFrame, df_tags: pd.DataFrame) -> None:
    # Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Carica il modello MF_SGD_User_Based
    model_path = "./models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl"
    recomm: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path)

    for fun in [linear_epsilon_decay]:
        logging.info(f"Funzione di decay epsilon: {fun.__name__}")

        dizionario = {"exploration": 0, "exploitation": 0, "comulative_reward": 0.0}
        for user_id in utility_matrix.index:
            exp, exploit, reward = test_single_user(user_id, recomm, utility_matrix, fun)
            dizionario["exploration"] += exp
            dizionario["exploitation"] += exploit
            dizionario["comulative_reward"] += reward

        evaluation_output = [
            f"Funzione epsilon decay: {fun.__name__}",
            f"Numero di esplorazioni: {dizionario['exploration']}",
            f"Numero di sfruttamenti: {dizionario['exploitation']}",
            f"Ricompensa totale: {dizionario['comulative_reward']}",
        ]
        with open(f"results/{fun.__name__}_mab.txt", "w") as f:
            f.write("\n" + "=" * 70 + "\n")
            for line in evaluation_output:
                f.write(line + "\n")
            f.write("=" * 70 + "\n")


def test_all_users(df_movies, df_ratings, df_tags):
    # Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Carica il modello MF_SGD_User_Based
    model_path = "./models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl"
    recomm: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path)

    epsilon = 1.0  # Epsilon iniziale
    dizionario = {"exploration": 0, "exploitation": 0, "comulative_reward": 0.0}
    for user_id in utility_matrix.index:
        exp, exploit, reward = test_single_user(user_id, recomm, utility_matrix, None, epsilon)
        dizionario["exploration"] += exp
        dizionario["exploitation"] += exploit
        dizionario["comulative_reward"] += reward

    evaluation_output = [
        f"Epsilon Mab: {epsilon}",
        f"Numero di esplorazioni: {dizionario['exploration']}",
        f"Numero di sfruttamenti: {dizionario['exploitation']}",
        f"Ricompensa totale: {dizionario['comulative_reward']}",
    ]
    with open(f"results/{epsilon}_mab.txt", "w") as f:
        f.write("\n" + "=" * 70 + "\n")
        for line in evaluation_output:
            f.write(line + "\n")
        f.write("=" * 70 + "\n")


if __name__ == "__main__":
    # Carica il dataset MovieLens
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")

    # test_dacys_all_users(df_movies, df_ratings, df_tags)

    test_all_users(df_movies, df_ratings, df_tags)
