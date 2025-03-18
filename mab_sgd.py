import numpy as np
import pandas as pd
from eps_mab import EpsGreedyMAB
from mf_sgd import MF_SGD_User_Based
from utils import load_movielens_data, min_max_normalize


def _print_info_rounds(i: int, curr_arm: int, curr_movie_id: int, curr_movie_title: str, reward: float) -> None:
    print(f"Round {i}:")
    print(f"  - Braccio selezionato: {curr_arm} -> MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
    print(f"  - Reward: {reward:.3f}\n")


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


def compute_reward(predicted_rating, selections):
    """Calcola la reward penalizzando i film selezionati troppe volte"""
    return predicted_rating / (1 + np.log(1 + selections))


def _start_rounds_mf_sgd(
    num_rounds: int,
    bandit_mab: EpsGreedyMAB,
    df_recommendations: pd.DataFrame,
) -> None:

    for i in range(0, num_rounds):

        # 0. Il bandit seleziona un braccio
        curr_arm: int = bandit_mab.play()

        # 1. Recupera il movieId e il titolo del film selezionato
        curr_movie_id: int = df_recommendations.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_recommendations.iloc[curr_arm]["title"]

        # 2. Calcola il reward per il braccio selezionato
        predicted_rating: float = df_recommendations.iloc[curr_arm]["predicted rating"]
        curr_arm_clicks: int = bandit_mab.get_clicks_for_arm()[curr_arm]
        # 3. Normalizzazione dei valori per la reward
        predicted_rating: float = min_max_normalize(predicted_rating, min_val=0.5, max_val=5.0)
        curr_arm_clicks: float = min_max_normalize(curr_arm_clicks, min(bandit_mab.get_clicks_for_arm()), max(bandit_mab.get_clicks_for_arm()))

        reward: float = compute_reward(predicted_rating, curr_arm_clicks)

        # Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_arm, reward)


def mab_on_sgd(df_ratings: pd.DataFrame, df_movies: pd.DataFrame, user_id: int, num_rounds: int = 1000, N: int = 20) -> list:
    # 1. Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # 2. Carica il modello MF-SGD
    model_path2 = "models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl"
    recomm: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path2)

    # 3. Raccomandazioni per l'utente user_id
    df_recommendations: pd.DataFrame = recomm.get_recommendations(utility_matrix, user_id)
    df_recommendations = df_recommendations.merge(df_movies, on="movieId")[["title", "values"]].head(N)
    df_recommendations.rename(columns={"values": "predicted rating"}, inplace=True)

    # 4.  Resetta l'indice per renderlo compatibile con il bandit
    df_recommendations.reset_index(drop=False, inplace=True)
    print(f"Reccomendations:\n {df_recommendations}")

    # 5. Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=N, epsilon=0.1, Q0=0.0)

    # Simulazione del gioco
    _start_rounds_mf_sgd(num_rounds, bandit_mab, df_recommendations)

    # Recupera i top k film raccomandati con il bandit
    return _get_topk_movies(bandit_mab, df_recommendations)


def main():
    # 0. Carica i dati di MovieLens (df_movies, df_ratings, df_tags)
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")
    print(f"Dataset caricato: {len(df_movies)} film, {len(df_ratings)} voti, {len(df_tags)} tag")

    # User to test
    temp_user_id = 1

    # Mab on SGD
    print(mab_on_sgd(df_ratings, df_movies, temp_user_id, num_rounds=1000, N=20))


if __name__ == "__main__":
    main()
