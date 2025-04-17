import logging
import numpy as np
import pandas as pd
from utils import min_max_normalize
from epsilon_mab import EpsGreedyMAB
from mf_sgd import MF_SGD_User_Based

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _print_info_rounds(i: int, curr_arm: int, curr_movie_id: int, curr_movie_title: str, reward: float) -> None:
    logging.info(f"Round {i}:")
    logging.info(f"  - Braccio selezionato: {curr_arm} -> MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
    logging.info(f"  - Reward: {reward:.3f}\n")


def _print_final_stats(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame) -> None:
    logging.info("\nStatistiche finali del bandit:")
    top_n_arms = bandit_mab.get_top_n()  # Restituisce i top N bracci con i relativi Q-values ordinati
    for i, (curr_arm, q_value) in enumerate(top_n_arms):
        curr_movie_id = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title = df_recommendations.iloc[curr_arm]["title"]

        logging.info(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"con Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_total_rewards_list()[curr_arm]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[curr_arm]} volte"
        )


def _get_topk_movies(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame) -> None:
    logging.info("\nTop film raccomandati:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, q_value) in enumerate(top_n_arms):
        curr_movie_id = df_recommendations.iloc[curr_selected_arm]["movieId"]
        topk.append(curr_movie_id)
    return topk


def compute_reward(predicted_rating, selections):
    """La reward è basata sulla predizione di rating del modello MF,
    ma penalizzata in base a quante volte il braccio (film) è già stato selezionato
    (selections, che corrisponde a curr_arm_clicks).
    La penalizzazione è logaritmica, quindi diminuisce man mano che il numero di selezioni aumenta."""
    return predicted_rating / (1 + np.log(1 + selections))


def _start_rounds_mf_sgd(
    num_rounds: int,
    bandit_mab: EpsGreedyMAB,
    df_recommendations: pd.DataFrame,
) -> None:

    for i in range(0, num_rounds):

        # Il bandit seleziona un braccio
        curr_arm: int = bandit_mab.play()

        # Recupera il movieId e il titolo del film selezionato
        curr_movie_id: int = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_recommendations.iloc[curr_arm]["title"]

        # Calcola il reward per il braccio selezionato
        predicted_rating: float = df_recommendations.iloc[curr_arm]["predicted rating"]
        curr_arm_clicks: int = bandit_mab.get_clicks_for_arm()[curr_arm]
        # Normalizzazione dei valori per la reward
        predicted_rating: float = min_max_normalize(predicted_rating, min_val=0.5, max_val=5.0)
        curr_arm_clicks: float = min_max_normalize(curr_arm_clicks, min(bandit_mab.get_clicks_for_arm()), max(bandit_mab.get_clicks_for_arm()))

        reward: float = compute_reward(predicted_rating, curr_arm_clicks)

        # Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_arm, reward)


def mab_on_sgd(df_ratings: pd.DataFrame, df_movies: pd.DataFrame, user_id: int, num_rounds: int = 1000, N: int = 20) -> list:
    """Simula il bandit su MF-SGD per raccomandare film all'utente user_id."""
    # Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Carica il modello MF-SGD
    model_path2 = "models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl"
    recomm: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path2)

    # Raccomandazioni per l'utente user_id
    df_recommendations: pd.DataFrame = recomm.get_recommendations(utility_matrix, user_id)
    df_recommendations = df_recommendations.merge(df_movies, on="movieId")[["title", "values"]].head(N)
    df_recommendations.rename(columns={"values": "predicted rating"}, inplace=True)

    # Resetta l'indice per renderlo compatibile con il bandit
    df_recommendations.reset_index(drop=False, inplace=True)
    logging.info(f"Reccomendations:\n {df_recommendations}")

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=N, epsilon=0.1, Q0=0.0)

    # Simulazione del gioco
    _start_rounds_mf_sgd(num_rounds, bandit_mab, df_recommendations)

    _print_final_stats(bandit_mab, df_recommendations)

    # Recupera i top k film raccomandati con il bandit
    return _get_topk_movies(bandit_mab, df_recommendations)
