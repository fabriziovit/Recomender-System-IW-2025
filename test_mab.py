import numpy as np
import pandas as pd
from utils import exp_epsilon_decay, linear_epsilon_decay, log_epsilon_decay
from epsilon_mab import EpsGreedyMAB
from mf_sgd import MF_SGD_User_Based
from utils import load_movielens_data


def _print_final_stats(df_merged: pd.DataFrame, bandit_mab: EpsGreedyMAB) -> None:
    print("\nStatistiche finali del bandit:")
    top_n_arms = bandit_mab.get_top_n()[0:20]  #! Restituisce i top N bracci con i relativi Q-values ordinati
    for i, (curr_arm, _) in enumerate(top_n_arms):
        curr_movie_id = df_merged.iloc[curr_arm]["movieId"]
        curr_movie_title = df_merged.iloc[curr_arm]["title"]

        print(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"con Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_total_rewards_list()[curr_arm]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[curr_arm]} volte"
        )
    print(f"num_exploration: {bandit_mab._nexploration}")
    print(f"num_exploitation: {bandit_mab._nexploitation}")


def _get_top_movies(bandit_mab: EpsGreedyMAB, df_merged: pd.DataFrame) -> list:
    print("\nTop film raccomandati:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, _) in enumerate(top_n_arms):
        curr_movie_id = df_merged.iloc[curr_selected_arm]["movieId"]
        topk.append(curr_movie_id)
    return topk


def _start_rounds(df_expected: pd.DataFrame, bandit_mab: EpsGreedyMAB, num_rounds: int) -> None:

    for i in range(0, num_rounds):

        # Il bandit seleziona un braccio (film)
        curr_arm: int = bandit_mab.play()

        # Recupero informazioni sul film selezionato
        curr_movie_id: int = df_expected.iloc[curr_arm]["movieId"]
        curr_movie_title: str = df_expected.iloc[curr_arm]["title"]
        print(f"\ncurr_selected_arm: {curr_arm}")
        print(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        # La reward corrisponde alla predizione del film selezionato
        reward = df_expected.iloc[curr_arm]["values"]

        '''
        print(f"Round {i}:")
        print(f"  - Braccio selezionato: {curr_arm} -> MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
        print(f"  - Reward: {reward:.3f}, epsilon: {bandit_mab.get_curr_epsilon():.3f}")
        print(f"  - function: {bandit_mab._epsilon_decay_function.__name__}")
        '''
        # Aggiorna epsilon ad ogni round
        bandit_mab.update_epsilon(i)

        # Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_arm, reward)


def mab(df_expected: pd.DataFrame, bandit_mab: EpsGreedyMAB, num_rounds: int = 10_000) -> list:

    # Simulazione del gioco
    _start_rounds(df_expected, bandit_mab, num_rounds)
    _print_final_stats(df_expected, bandit_mab)

    # Recupera i top k film raccomandati con il bandit
    return _get_top_movies(bandit_mab, df_expected)


def main():
    # Carica il dataset MovieLens
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")

    # Crea la utility matrix
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Carica il modello MF_SGD_User_Based
    model_path = "./models/mf_model_n200_lr0.001_lambda0.0001_norm.pkl"
    recomm: MF_SGD_User_Based = MF_SGD_User_Based.load_model(model_path)

    # Utente da testare
    user_id = 1

    # Ottieni le predizioni per l'utente specificato user_id
    df_expected = recomm.get_recommendations(utility_matrix, user_id)
    df_expected = df_expected.merge(df_movies, on="movieId")[["title", "values"]].sort_index(ascending=True)
    # Imposta l'indice per renderlo compatibile con il bandit
    new_index = pd.RangeIndex(df_expected.shape[0], name="idxarm")
    df_expected.reset_index(drop=False, inplace=True)
    df_expected.set_index(new_index, inplace=True)
    print(f"Dimensioni df_expected: {df_expected.shape}")
    print(f"df_expectet):\n{df_expected}")

    ret = {}
    for fun in [exp_epsilon_decay, log_epsilon_decay, linear_epsilon_decay]:
        print(f"Funzione di decay epsilon: {fun.__name__}")
        bandit_mab = EpsGreedyMAB(n_arms=df_expected.shape[0], epsilon=0.99, Q0=0.0)
        bandit_mab.set_epsilon_deacy(fun)

        top_k = mab(df_expected, bandit_mab, num_rounds=10_000)[:10]  #! Restituisce i primi 10 film raccomandati

        # Salvo i risultati in un dizionario
        ret[fun.__name__] = (top_k, ("nexploration: ", bandit_mab._nexploration), ("nexploitation: ", bandit_mab._nexploitation))

    print(ret)


if __name__ == "__main__":
    main()
