import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import argparse

# Import dei recommender systems
try:
    from cb_recommender import ContentBasedRecommender
except ImportError:
    print("Warning: ContentBasedRecommender non trovato. Alcune funzionalità non saranno disponibili.")

try:
    from cf_recommender import CollaborativeRecommender, get_ratings_mean
    from sklearn.neighbors import NearestNeighbors
    from utils import load_movielens_data, pearson_distance
except ImportError:
    print("Warning: CollaborativeRecommender o dipendenze non trovate. Alcune funzionalità non saranno disponibili.")

try:
    from director_recommender import recommend_by_movie_id, recommend_films_with_actors
except ImportError:
    print("Warning: Director recommender non trovato. Alcune funzionalità non saranno disponibili.")


class MovieRecommenderApp:
    """
    Applicazione con interfaccia a riga di comando per utilizzare diversi sistemi di raccomandazione per film.
    Supporta:
    - Raccomandazioni content-based (basate su abstract e generi)
    - Raccomandazioni collaborative (user-based e item-based)
    - Raccomandazioni basate sul regista (usando DBpedia e Wikidata)
    """

    def __init__(self, data_dir: str = "datasets/"):
        """
        Inizializza l'applicazione caricando i dati necessari.
        
        Args:
            data_dir: Directory contenente i dataset MovieLens
        """
        self.data_dir = data_dir
        self.df_movies = None
        self.df_ratings = None
        self.df_tags = None
        self.content_recommender = None
        self.collaborative_recommender = None
        self.utility_matrix = None
        
        # Flag per indicare quali componenti sono stati inizializzati
        self.content_initialized = False
        self.collaborative_initialized = False
        self.director_initialized = False
        
        # Path al file CSV per le raccomandazioni basate sul regista
        self.movies_with_abstracts_path = "C:/Users/Fabrizio/Documents/Progetto IW/Recomender-System-IW-2025/recommender_systems/datasets/movies_with_abstracts_complete.csv" #! Cambiare path con uno dinamico
        #self.movies_with_abstracts_path = "C:/Users/Fabrizio/Documents/Progetto IW/Recomender-System-IW-2025/recommender_systems/datasets/movies_with_abstracts_complete.csv"
        self._load_basic_data()

    def _load_basic_data(self) -> None:
        """Carica i dati base di MovieLens se la funzione è disponibile"""
        try:
            self.df_movies, self.df_ratings, self.df_tags = load_movielens_data(self.data_dir)
            print(f"Dati caricati: {len(self.df_movies)} film, {len(self.df_ratings)} valutazioni")
            
            # Crea la matrice utenti-film pivot (userId x movieId)
            self.utility_matrix = self.df_ratings.pivot(index="userId", 
                                                  columns="movieId", 
                                                  values="rating").fillna(0)
            print(f"Matrice utility creata: {self.utility_matrix.shape}")
        except Exception as e:
            print(f"Errore nel caricamento dei dati base: {e}")

    def initialize_content_recommender(self) -> None:
        """Inizializza il recommender basato sul contenuto se non è già stato inizializzato"""
        if self.content_initialized:
            print("Content-based recommender già inizializzato.")
            return
            
        try:
            # Carica i dati con gli abstract se non sono già stati caricati
            df_with_abstracts = pd.read_csv(self.movies_with_abstracts_path)
            print(f"Dati con abstract caricati: {len(df_with_abstracts)} film")
            
            # Inizializza il ContentBasedRecommender
            self.content_recommender = ContentBasedRecommender(df_with_abstracts)
            self.content_initialized = True
            print("Content-based recommender inizializzato con successo.")
        except Exception as e:
            print(f"Errore nell'inizializzazione del content recommender: {e}")
    
    def initialize_collaborative_recommender(self, n_neighbors: int = 10) -> None:
        """Inizializza il recommender collaborativo se non è già stato inizializzato"""
        if self.collaborative_initialized:
            print("Collaborative recommender già inizializzato.")
            return
            
        try:
            # Inizializza i modelli NearestNeighbors
            knn_model_item = NearestNeighbors(metric=pearson_distance, 
                                             algorithm="brute", 
                                             n_neighbors=n_neighbors, 
                                             n_jobs=-1)
            
            knn_model_user = NearestNeighbors(metric=pearson_distance, 
                                             algorithm="brute", 
                                             n_neighbors=n_neighbors, 
                                             n_jobs=-1)
            
            # Inizializza il CollaborativeRecommender
            self.collaborative_recommender = CollaborativeRecommender(knn_model_item, knn_model_user)
            
            # Addestra i modelli
            print("Addestramento modello item-based...")
            self.collaborative_recommender.fit_item_model(self.utility_matrix, re_fit=True)
            
            print("Addestramento modello user-based...")
            self.collaborative_recommender.fit_user_model(self.utility_matrix, re_fit=True)
            
            self.collaborative_initialized = True
            print("Collaborative recommender inizializzato con successo.")
        except Exception as e:
            print(f"Errore nell'inizializzazione del collaborative recommender: {e}")

    def check_director_recommender(self) -> bool:
        """Verifica che il file necessario per le raccomandazioni basate sul regista esista"""
        if os.path.exists(self.movies_with_abstracts_path):
            self.df_with_abstracts = pd.read_csv(self.movies_with_abstracts_path)
            self.director_initialized = True
            return True
        else:
            print(f"File {self.movies_with_abstracts_path} non trovato.")
            print("Le raccomandazioni basate sul regista non saranno disponibili.")
            return False

    def print_menu(self) -> None:
        """Stampa il menu principale dell'applicazione"""
        print("\n" + "=" * 50)
        print("      SISTEMA DI RACCOMANDAZIONE FILM")
        print("=" * 50)
        print("1. Raccomandazioni basate sul contenuto (abstract e generi)")
        print("2. Raccomandazioni collaborative item-based")
        print("3. Raccomandazioni collaborative user-based")
        print("4. Raccomandazioni basate sul regista")
        print("5. Cerca un film per titolo")
        print("6. Visualizza dettagli film per ID")
        print("0. Esci")
        print("=" * 50)

    def search_movie_by_title(self, query: str) -> List[Dict[str, Any]]:
        """Cerca un film per titolo"""
        if self.df_movies is None:
            print("Dati non caricati.")
            return []
            
        # Esegue la ricerca case-insensitive
        results = self.df_movies[self.df_movies['title'].str.contains(query, case=False)]
        
        if len(results) == 0:
            print(f"Nessun film trovato con la query '{query}'.")
            return []
            
        # Converti i risultati in lista di dizionari
        movies_list = []
        for _, row in results.iterrows():
            movies_list.append({
                'id': row.name,
                'title': row['title'],
                'genres': row['genres']
            })
            
        return movies_list
        
    def show_movie_details(self, movie_id: int) -> None:
        """Mostra i dettagli di un film specifico"""
        if self.df_movies is None:
            print("Dati non caricati.")
            return
            
        try:
            movie = self.df_movies.loc[movie_id].to_dict()
            #print(f"movie {movie}")
            print("\nDettagli film:")
            print(f"ID: {movie_id}")
            print(f"Titolo: {movie['title']}")
            print(f"Generi: {movie['genres']}")
            
            # Mostra valutazione media se disponibile
            if self.df_ratings is not None:
                ratings = self.df_ratings[self.df_ratings['movieId'] == movie_id]
                if len(ratings) > 0:
                    avg_rating = ratings['rating'].mean()
                    num_ratings = len(ratings)
                    print(f"Valutazione media: {avg_rating:.2f} (basata su {num_ratings} valutazioni)")
        except (IndexError, KeyError):
            print(f"Film con ID {movie_id} non trovato.")
            
    def run_content_recommender(self) -> None:
        """Gestisce le raccomandazioni basate sul contenuto"""
        if not self.content_initialized:
            self.initialize_content_recommender()
            if not self.content_initialized:
                print("Non è possibile utilizzare il content-based recommender. Componente non inizializzato.")
                return
        
        query = input("\nInserisci il titolo del film (anche parziale) per cui cercare raccomandazioni: ")
        results = self.search_movie_by_title(query)
        
        if not results:
            return
            
        print("\nFilm trovati:")
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']} (ID: {movie['id']})")
            
        try:
            choice = int(input("\nSeleziona il numero del film (o 0 per tornare al menu): "))
            if choice == 0 or choice > len(results):
                return
                
            selected_movie = results[choice-1]
            movie_title = selected_movie['title']
            
            # Ottieni raccomandazioni
            top_n = int(input("Quante raccomandazioni vuoi visualizzare? [default: 10] ") or 10)
            recommended = self.content_recommender.recommend(movie_title, top_n=top_n)
            
            print(f"\nFilm consigliati simili a '{movie_title}':")
            for i, (_, row) in enumerate(recommended.iterrows(), 1):
                print(f"{i}. {row['title']} - (ID: {row.name}) - Generi: {row['genres']}")
                
        except (ValueError, IndexError) as e:
            print(f"Errore nella selezione: {e}")

    def run_collaborative_item_recommender(self) -> None:
        """Gestisce le raccomandazioni collaborative basate sugli item"""
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                print("Non è possibile utilizzare il collaborative recommender. Componente non inizializzato.")
                return
        
        query = input("\nInserisci il titolo del film (anche parziale) per cui cercare raccomandazioni: ")
        results = self.search_movie_by_title(query)
        
        if not results:
            return
            
        print("\nFilm trovati:")
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']} (ID: {movie['id']})")
            
        try:
            choice = int(input("\nSeleziona il numero del film (o 0 per tornare al menu): "))
            if choice == 0 or choice > len(results):
                return
                
            selected_movie = results[choice-1]
            movie_id = selected_movie['id']
            
            # Ottieni raccomandazioni
            top_n = int(input("Quante raccomandazioni vuoi visualizzare? [default: 10] ") or 10)
            recommended = self.collaborative_recommender.get_item_recommendations(
                movie_id, 
                self.utility_matrix, 
                self.df_movies
            )
            
            print(f"\nFilm consigliati simili a '{selected_movie['title']}' (item-based):")
            for i, (_, row) in enumerate(recommended.head(top_n).iterrows(), 1):
                print(f"{i}. {row['title']} - (ID: {row.name}) - Generi: {row['genres']}")
                
        except (ValueError, IndexError) as e:
            print(f"Errore nella selezione: {e}")
        except Exception as e:
            print(f"Errore durante la raccomandazione: {e}")

    def run_collaborative_user_recommender(self) -> None:
        """Gestisce le raccomandazioni collaborative basate sugli utenti"""
        if not self.collaborative_initialized:
            self.initialize_collaborative_recommender()
            if not self.collaborative_initialized:
                print("Non è possibile utilizzare il collaborative recommender. Componente non inizializzato.")
                return
        
        try:
            # Mostra alcuni utenti disponibili
            users_sample = sorted(np.random.choice(self.utility_matrix.index, 
                                             size=min(10, len(self.utility_matrix.index)), 
                                             replace=False).tolist())
            
            print("\nAlcuni ID utente disponibili (esempio):")
            print(", ".join(map(str, users_sample)))
            
            user_id = int(input("\nInserisci l'ID dell'utente per cui cercare raccomandazioni (1-610): "))
            
            if user_id not in self.utility_matrix.index:
                print(f"Utente con ID {user_id} non trovato nella matrice.")
                return
                
            # Ottieni raccomandazioni
            top_n = int(input("Quante raccomandazioni vuoi visualizzare? [default: 10] ") or 10)
            recommended = self.collaborative_recommender.get_user_recommendations(
                user_id, 
                self.utility_matrix, 
                self.df_movies
            )
            
            print(f"\nFilm consigliati per l'utente {user_id} (user-based):")
            for i, (_, row) in enumerate(recommended.head(top_n).iterrows(), 1):
                print(f"{i}. {row['title']} - (ID: {row.name}) - Valore previsto: {row['values']:.2f}")
                
        except ValueError as e:
            print(f"Errore nell'input: {e}")
        except Exception as e:
            print(f"Errore durante la raccomandazione: {e}")

    def run_director_recommender(self) -> None:
        """Gestisce le raccomandazioni basate sul regista"""
        if not self.check_director_recommender():
            return
            
        choice = input("\nVuoi cercare per (1) titolo del film o (2) nome regista? [1/2]: ")
        
        if choice == "1":
            try:
                query = input("Inserisci il titolo del film: ")
                results = self.search_movie_by_title(query)
                if results:
                    print("\nFilm trovati:")
                    for i, movie in enumerate(results, 1):
                        print(f"{i}. {movie['title']} (ID: {movie['id']}) - Generi: {movie['genres']}")
                    
                choice = int(input("\nSeleziona il numero del film (o 0 per tornare al menu): "))
                if choice == 0 or choice > len(results):
                    return
                    
                selected_movie = results[choice-1]
                movie_title = selected_movie['title']
                movie_id = selected_movie['id']

                max_actors = int(input("Numero massimo di attori da mostrare per film [default: 5]: ") or 5)
                recommend_by_movie_id(self.movies_with_abstracts_path, movie_id, max_actors=max_actors)
            except ValueError as e:
                print(f"Errore nell'input: {e}")
        
        elif choice == "2":
            director = input("Inserisci il nome del regista (o premi invio per Christopher Nolan): ")
            if not director:
                director = "Christopher Nolan"
            for _, row in self.df_with_abstracts.iterrows():
                if row["dbpedia_director"] == director:
                    movie_title = row['title']
                    break
                
            max_actors = int(input("Numero massimo di attori da mostrare per film [default: 5]: ") or 5)
            recommend_films_with_actors(director, max_actors=max_actors, title=movie_title)
            
        else:
            print("Scelta non valida.")

    def run(self) -> None:
        """Esegue l'applicazione principale con il menu interattivo"""
        print("\nBenvenuto nel sistema di raccomandazione film!")
        print("Inizializzazione dell'applicazione...")
        
        while True:
            self.print_menu()
            choice = input("Seleziona un'opzione: ")
            
            if choice == "0":
                print("Grazie per aver utilizzato il sistema di raccomandazione film!")
                break
                
            elif choice == "1":
                self.run_content_recommender()
                
            elif choice == "2":
                self.run_collaborative_item_recommender()
                
            elif choice == "3":
                self.run_collaborative_user_recommender()
                
            elif choice == "4":
                self.run_director_recommender()
                
            elif choice == "5":
                query = input("\nInserisci il titolo del film da cercare: ")
                results = self.search_movie_by_title(query)
                
                if results:
                    print("\nFilm trovati:")
                    for i, movie in enumerate(results, 1):
                        print(f"{i}. {movie['title']} (ID: {movie['id']}) - Generi: {movie['genres']}")
                
            elif choice == "6":
                try:
                    movie_id = int(input("\nInserisci l'ID del film: "))
                    self.show_movie_details(movie_id)
                except ValueError:
                    print("ID non valido. Inserisci un numero intero.")
                    
            else:
                print("Opzione non valida. Riprova.")
                
            input("\nPremi Invio per continuare...")


def main():
    parser = argparse.ArgumentParser(description="Sistema di raccomandazione film")
    parser.add_argument('--data_dir', type=str, default='./dataset/', 
                        help='Directory contenente i dataset MovieLens')
    args = parser.parse_args()
    
    app = MovieRecommenderApp(data_dir=args.data_dir)
    app.run()


if __name__ == "__main__":
    main()