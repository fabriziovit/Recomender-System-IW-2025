from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from cb_recommender import ContentBasedRecommender
from cf_recommender import CollaborativeRecommender
from eps_mab import EpsGreedyMAB
from utils import load_movielens_data, pearson_distance, compute_mean_form_movie, compute_hybrid_reward


def _print_info_rounds(i: int, curr_arm: int, curr_idx: Optional[int], curr_movie_id: int, curr_movie_title: str, curr_sim: float, curr_mean: float, reward: float) -> None:
    print(f"Round {i}:")
    if curr_idx:
        print(f"  - Braccio selezionato: {curr_arm} -> Index: {curr_idx}, MovieId: {curr_movie_id}, titolo: {curr_movie_title}")
    else:
        print(f"  - Braccio selezionato: {curr_arm} -> MovieId: {curr_movie_id}, titolo: {curr_movie_title}")

    print(f"  - Similarità: {curr_sim:.3f}, Mean Normalizzata: {curr_mean:.3f}, Hybrid reward: {reward:.3f}")
    print()


def _start_rounds(
    num_rounds: int,
    bandit_mab: EpsGreedyMAB,
    df_recommendations: pd.DataFrame,
    indexes_embedd_of_similiar: pd.Index,
    sim_scores: np.ndarray,
    df_ratings: pd.DataFrame,
    collab_filter: bool = False,
) -> None:

    for i in range(0, num_rounds):

        # 0. Il bandit seleziona un braccio
        curr_selected_arm: int = bandit_mab.play()
        #print(f"curr_selected_arm: {curr_selected_arm}")
        if not collab_filter:
            # Content-Based: Recupera l'indice dell'embedding del film selezionato dal bandit
            curr_idx_embedd: int = indexes_embedd_of_similiar[curr_selected_arm]
            # Recupera il movieId e il titolo del film selezionato
            curr_movie_id: int = df_recommendations.loc[curr_idx_embedd]["movieId"]
            curr_movie_title: str = df_recommendations.loc[curr_idx_embedd]["title"]
        else:
            # Collaborative: Recupera l'indice dell'embedding del film selezionato dal bandit
            curr_movie_id: int = df_recommendations.iloc[curr_selected_arm]["movieId"]
            curr_movie_title: str = df_recommendations.iloc[curr_selected_arm]["title"]

        #print(f"curr_movie_id: {curr_movie_id}, curr_movie_title: {curr_movie_title}")

        # 1. Ottieni il punteggio di similarità per il film selezionato (dal vettore sim_scores)
        curr_similarity: float = sim_scores[curr_selected_arm]

        # 2. Calcola la reward media normalizzata per il film selezionato
        curr_mean_reward: float = compute_mean_form_movie(curr_movie_id, df_ratings)

        # 3. Calcola la hybrid reward
        reward = compute_hybrid_reward(curr_similarity, curr_mean_reward, beta=0.8)

        '''
        if not collab_filter:
            _print_info_rounds(i, curr_selected_arm, curr_idx_embedd, curr_movie_id, curr_movie_title, curr_similarity, curr_mean_reward, reward)
        else:
            _print_info_rounds(i, curr_selected_arm, None, curr_movie_id, curr_movie_title, curr_similarity, curr_mean_reward, reward)
        '''
        # 4. Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_selected_arm, reward)


def _print_final_stats(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index, collab_filter: bool = False) -> None:
    print("\nStatistiche finali del bandit:")
    top_n_arms = bandit_mab.get_top_n()  # Restituisce i top N bracci con i relativi Q-values ordinati
    for i, (curr_arm, q_value) in enumerate(top_n_arms):
        if not collab_filter:
            curr_idx_embedd = indexes_of_embedd[curr_arm]
            curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        else:
            curr_movie_id = df_recommendations.iloc[curr_arm]["movieId"]
            curr_movie_title = df_recommendations.iloc[curr_arm]["title"]
        print(
            f"  - Arm {curr_arm}: (Movie ID {curr_movie_id}, '{curr_movie_title}') "
            f"con Q = {bandit_mab.get_qvalues()[curr_arm]:.2f}, reward_tot = {bandit_mab.get_rewards_list()[curr_arm]:.2f}"
            f" e selezionato {bandit_mab.get_clicks_for_arm()[curr_arm]} volte"
        )


def _get_topk_movies(bandit_mab: EpsGreedyMAB, df_recommendations: pd.DataFrame, indexes_of_embedd: pd.Index, collab_filter: bool = False) -> None:
    print("\nTop film raccomandati:")
    top_n_arms = bandit_mab.get_top_n()
    topk = []
    for i, (curr_selected_arm, q_value) in enumerate(top_n_arms):
        if not collab_filter:
            curr_idx_embedd = indexes_of_embedd[curr_selected_arm]
            curr_movie_id = df_recommendations.loc[curr_idx_embedd]["movieId"]
        else:
            curr_movie_id = df_recommendations.iloc[curr_selected_arm]["movieId"]
        topk.append(curr_movie_id)
    return topk


def mab_on_contentbased(movie_title: str, df_ratings: pd.DataFrame, N: int) -> None:

    # 0. Carica il dataset per il content-based recommender
    df = pd.read_csv("dataset/movies_with_abstracts_complete.csv", on_bad_lines="warn")
    recommender = ContentBasedRecommender(df, abstract_col="dbpedia_abstract", title_col="title", genres_col="genres")

    # 1. Ottieni l'indice del film selezionato
    curr_movie_id = recommender.df[recommender.df["title"] == movie_title]["movieId"]
    curr_idx_embedd = recommender.get_idx(movie_title)

    # 2. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recommender.recommend(movie_title, N)[["movieId", "title"]]
    indexes_of_embedd: pd.Index = df_recommendations.index

    # 3. Calcola i punteggi di similarità tra il film e quelli raccomandati
    sim_scores_items: np.ndarray = recommender.compute_similarity_scores(curr_idx_embedd)
    sim_scores_items: np.ndarray = sim_scores_items[indexes_of_embedd]

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=N, epsilon=0.1, Q0=0.0)

    # Simulazione del gioco
    num_rounds = 10_000
    _start_rounds(num_rounds, bandit_mab, df_recommendations, indexes_of_embedd, sim_scores_items, df_ratings)

    return _get_topk_movies(bandit_mab, df_recommendations, indexes_of_embedd)


def _mab_on_collabfilter_item(recomm: CollaborativeRecommender, movie_id: int, df_ratings: pd.DataFrame, df_movies: pd.DataFrame, NN) -> None:

    # 2. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recomm.get_item_recommendations(movie_id, df_movies).head(NN)
    recomm_movie_ids: pd.Index = df_recommendations.index

    # 3. Recupero la similarità tra movie_id e recomm_movie_ids
    sim_scores = recomm._sim_items
    sim_scores = sim_scores[recomm_movie_ids].to_numpy()

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=NN, epsilon=0.1, Q0=0.0)

    # Resetta l'indice del DataFrame delle raccomandazioni per renderlo compatibile con il bandit
    df_recommendations.reset_index(drop=False, inplace=True)

    # Simulazione del gioco
    num_round = 10_000
    _start_rounds(num_round, bandit_mab, df_recommendations, recomm_movie_ids, sim_scores, df_ratings, collab_filter=True)

    # Stampa le statistiche finali del bandit
    _print_final_stats(bandit_mab, df_recommendations, recomm_movie_ids, collab_filter=True)

    # Recupera i top k film raccomandati con il bandit
    return _get_topk_movies(bandit_mab, df_recommendations, recomm_movie_ids, collab_filter=True)


def _mab_on_collabfilter_user(recomm: CollaborativeRecommender, matrix: pd.DataFrame, user_id: int, df_ratings: pd.DataFrame, df_movies: pd.DataFrame, NN) -> None:

    # 2. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recomm.get_user_recommendations(user_id, matrix, df_movies).head(NN)
    recomm_movie_ids: pd.Index = df_recommendations.index

    # 3. Recupero la similarità tra user e similiar_user
    # !NOTA: Solo in questo caso, per il calcolo della hybrid reward si utilizza la similarità tra l'utente e i simili
    # !beta * similarity + (1 - beta) * mean_reward: dove similarity è la similarità tra l'utente e i simili
    sim_scores_users = recomm._sim_users.to_numpy()

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=NN, epsilon=0.1, Q0=0.0)

    # Resetta l'indice del DataFrame delle raccomandazioni per renderlo compatibile con il bandit
    df_recommendations.reset_index(drop=False, inplace=True)

    # Simulazione del gioco
    num_round = 10_000
    _start_rounds(num_round, bandit_mab, df_recommendations, recomm_movie_ids, sim_scores_users, df_ratings, collab_filter=True)

    # Stampa le statistiche finali del bandit
    _print_final_stats(bandit_mab, df_recommendations, recomm_movie_ids, collab_filter=True)

    # Recupera i top k film raccomandati con il bandit
    return _get_topk_movies(bandit_mab, df_recommendations, recomm_movie_ids, collab_filter=True)


def mab_on_collabfilter(df_ratings: pd.DataFrame, df_movies: pd.DataFrame, movie_id: Optional[int] = None, user_id: Optional[int] = None, N=20):
    # Crea la matrice utenti-film pivot (userId x movieId)
    utility_matrix = df_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"# Numero di utenti totali: {len(utility_matrix.index)}")
    print(f"# Numero di movies totali: {len(utility_matrix.columns)}")
    print(f"# Utility-matrix.shape: {utility_matrix.shape}")

    # Inizializza il modello NearestNeighbors con metrica di correlazione di Pearson
    knn_model_pearson_item = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=N + 1, n_jobs=-1)
    knn_model_pearson_user = NearestNeighbors(metric=pearson_distance, algorithm="brute", n_neighbors=N + 1, n_jobs=-1)

    # Istanzia il Recommender con il modello KNN
    recomm = CollaborativeRecommender(knn_model_pearson_item, knn_model_pearson_user)
    recomm.fit_user_model(utility_matrix, re_fit=True)  # Addestra il modello user-based
    recomm.fit_item_model(utility_matrix, re_fit=True)  # Addestra il modello item-based

    if not movie_id and not user_id:
        raise ValueError("Almeno movie_id o user_id devono essere specificati")

    if movie_id and user_id:
        # Raccomandazioni per entrambi
        recomm_mab_item = _mab_on_collabfilter_item(recomm, movie_id, df_ratings, df_movies, N)
        recomm_mab_user = _mab_on_collabfilter_user(recomm, utility_matrix, user_id, df_ratings, df_movies, N)
        return recomm_mab_item, recomm_mab_user
    elif movie_id and not user_id:
        # Raccomandazioni per Item-Colaborative Filtering
        recomm_mab_item = _mab_on_collabfilter_item(recomm, movie_id, df_ratings, df_movies, N)
        return recomm_mab_item
    else:
        # Raccomandazioni per User-Colaborative Filtering
        recomm_mab_user = _mab_on_collabfilter_user(recomm, utility_matrix, user_id, df_ratings, df_movies, N)
        return recomm_mab_user

def main():
    # 0. Carica i dati di MovieLens (df_movies, df_ratings, df_tags)
    df_movies, df_ratings, df_tags = load_movielens_data("dataset/")
    print(f"Dataset caricato: {len(df_movies)} film, {len(df_ratings)} voti, {len(df_tags)} tag")

    #! User to test
    temp_user_id = 1
    #! Movie to test
    temp_movie_title = "Toy Story 2 (1999)"
    temp_movie_id = df_movies[df_movies["title"] == temp_movie_title].index[0]

    # MAB on content-based
    # print(mab_on_contentbased(temp_movie_title, df_ratings, N=20))

    # MAB on collaborative-filtering:
    # Controllare perché restituisce sempre i risultati nello stesso ordine!
    # Probabile che ci sia un problema con il calcolo della hybrid reward
    # Item-based collaborative filtering
    print(mab_on_collabfilter(df_ratings, df_movies, temp_movie_id, None, N=20))
    # User-based collaborative filtering
    print(mab_on_collabfilter(df_ratings, df_movies, None, temp_user_id, N=20))


if __name__ == "__main__":
    main()
