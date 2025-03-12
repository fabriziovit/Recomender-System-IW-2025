import numpy as np
import pandas as pd
from cb_recommender import ContentBasedRecommender
from eps_mab import EpsGreedyMAB
from utils import load_movielens_data


def get_movie_mean_reward(movie_id: str, df_ratings: pd.DataFrame) -> float:
    """
    Calcola il rating medio per il film e lo normalizza su scala [0,1] (scala originale 0-5).
    Se non sono presenti rating, restituisce 0.
    """
    min_rating_value: float = 0.0
    max_rating_value: float = 5.0
    movie_ratings = df_ratings[df_ratings["movieId"] == movie_id]["rating"]

    if movie_ratings.empty:
        return 0.0  # Se non ci sono rating, restituisce 0

    # Calcola il rating medio per il film
    avg_rating = movie_ratings.mean()

    # Normalizza il rating  [0,5] -> [0,1]
    return (avg_rating - min_rating_value) / (max_rating_value - min_rating_value)


def simulate_interactions():
    # 0. Carica i dati di MovieLens (df_movies, df_ratings, df_tags)
    _, df_ratings, _ = load_movielens_data("dataset/")

    # 1. Carica il dataset per il content-based recommender
    df = pd.read_csv("dataset/movies_with_abstracts_complete.csv", on_bad_lines="warn")
    recommender = ContentBasedRecommender(df, abstract_col="dbpedia_abstract", title_col="title", genres_col="genres")

    embeddings_matrix = recommender.get_embeddings()
    print(f"Shape di embeddings_matrix: {embeddings_matrix.shape}")

    temp_movie_id = "Toy Story 2 (1999)"
    temp_id = recommender.get_idx(movie_title=temp_movie_id)

    # 2. Otteniamo il DataFrame dei film raccomandati
    df_recommendations: pd.DataFrame = recommender.recommend(temp_movie_id)
    recomm_movie_ids: pd.Index = df_recommendations.index  # Recupera gli indici dei film raccomandati nel DataFrame completo
    print(f"\nFilm Raccomandati:\n{df_recommendations[['movieId','title']]}, con indici {recomm_movie_ids.tolist()}\n")

    # 3. Estrae gli embeddings per i film selezionati (context_matrix di forma (n_arms, n_dims))
    context_matrix_embeddings = embeddings_matrix[recomm_movie_ids, :]
    print(f"Shape of context_matrix: {context_matrix_embeddings.shape}\n")

    # 4. Calcola i punteggi di similarità solo per i film raccomandati
    sim_scores = recommender.compute_similarity_scores(temp_id)
    sim_scores = sim_scores[recomm_movie_ids]
    print(f"Similarity scores: {sim_scores}\n")

    # Istanziazione del bandit Epislon-Greedy MAB
    bandit_mab = EpsGreedyMAB(n_arms=10, n_dims=embeddings_matrix.shape[1], epsilon=0.1, Q0=0.0)

    num_rounds = 100
    for i in range(num_rounds):

        curr_selected_arm: int = bandit_mab.play(context_matrix_embeddings)

        curr_movie_df: pd.DataFrame = recomm_movie_ids[curr_selected_arm]
        curr_movie_id: int = df_recommendations.loc[curr_movie_df]["movieId"]
        curr_movie_title: str = df_recommendations.loc[curr_movie_df]["title"]

        # Ottieni il punteggio di similarità per il film selezionato (dal vettore sim_scores)
        curr_similarity: float = sim_scores[curr_selected_arm]

        # Calcola la reward media normalizzata per il film selezionato
        curr_mean_reward: float = get_movie_mean_reward(curr_movie_id, df_ratings)
        print(f"Mean reward: {curr_mean_reward:.3f}")

        # Calcola della reward: combinazione lineare di similarità e rating medio del film selezionato
        beta = 0.8  #! Peso per la componente di similarità
        reward = beta * curr_similarity + (1 - beta) * curr_mean_reward

        print(f"Round {i}:")
        print(f"  Braccio selezionato: {curr_selected_arm} -> MovieID: {curr_movie_id}, titolo: {curr_movie_title}")
        print(f"  Similarità: {curr_similarity:.3f}, Rating normalizzato: {curr_mean_reward:.3f}")
        print(f"  Hybrid reward: {reward:.3f}\n")

        # Aggiorna il bandit con la reward calcolata
        bandit_mab.update(curr_selected_arm, reward, context_matrix_embeddings)

    print("Statistiche finali del bandit:")
    for i in range(bandit_mab.get_narms()):
        curr_movie_df = recomm_movie_ids[i]
        curr_movie_id = df_recommendations.loc[curr_movie_df]["movieId"]
        curr_movie_title = df_recommendations.loc[curr_movie_df]["title"]
        print(
            f"Arm {i} (Movie ID {curr_movie_id}, '{curr_movie_title}'): " f"con Q = {bandit_mab.get_qvalues()[i]:.2f}, " f"e selezionato {bandit_mab.get_clicks_for_arm()[i]} volte"
        )

    print("\nTop film raccomandati:")
    top_n_list = bandit_mab.get_top_n()
    for i, (curr_selected_arm, q_value) in enumerate(top_n_list):
        temp_id = recomm_movie_ids[curr_selected_arm]
        curr_movie_id = df.loc[temp_id]["movieId"]
        curr_movie_title = df.loc[temp_id]["title"]
        print(
            f"Top {i+1}: Arm {curr_selected_arm} (Movie ID {curr_movie_id}, '{curr_movie_title}'): "
            f"con Q = {q_value:.2f}, "
            f"e selezionato {bandit_mab.get_clicks_for_arm()[curr_selected_arm]} volte"
        )


if __name__ == "__main__":
    simulate_interactions()
