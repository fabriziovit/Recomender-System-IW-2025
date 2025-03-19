import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


class ContentBasedRecommender:
    """
    Classe per un sistema di raccomandazione content-based.
    Questo sistema utilizza gli abstract dei film e i generi per fornire raccomandazioni.
    Sfrutta Word2Vec per la rappresentazione semantica degli abstract e la similarità del coseno
    per confrontare gli abstract. Per i generi, utilizza la similarità di Jaccard.
    Le similarità di abstract e generi sono combinate per generare un punteggio finale.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
    ):
        """
        Inizializza il ContentBasedRecommender.
        """
        self.df = df

        self._preprocess_all(vector_size=vector_size, window=window, min_count=min_count)

        print(f"Vocabolario Word2Vec creato con {len(self.model.wv.index_to_key)} parole uniche.")

    def _preprocess_text(self) -> None:
        """
        Preprocessa il testo degli abstract utilizzando Gensim.
        Effettua le seguenti operazioni:
        1. Tokenizzazione e normalizzazione: usa simple_preprocess (minuscolo, rimuove punteggiatura e tokenizza).
        2. Rimozione delle stopwords predefinite di Gensim.
        """
        self.tokenized_abstracts = []
        # Lista di liste, dove ogni lista interna contiene i tokens di un certo abstract
        self.tokenized_abstracts = [[word for word in simple_preprocess(text) if word not in STOPWORDS] for text in self.df["dbpedia_abstract"]]

    def _calculate_embeddings(self) -> None:
        """
        Calcola gli embedding di ogni abstract come media dei vettori Word2Vec.
        """
        embeddings = []
        for curr_tokens in self.tokenized_abstracts:
            word_vectors = [self.model.wv[word] for word in curr_tokens if word in self.model.wv]  # Recupero gli embeddings per ogni parola di tokens

            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
                assert doc_vector.shape[0] == self.model.vector_size, f"Errore: dimensione {doc_vector.shape[0]} invece di {self.model.vector_size}"
            else:
                doc_vector = np.zeros(self.model.vector_size)
            embeddings.append(doc_vector)
        print("End of _calculate_embeddings")
        self.embeddings = np.array(embeddings)

    def _parse_genres(self) -> None:
        """
        Parsifica la colonna dei generi trasformando le stringhe in set di generi.
        I generi sono separati dal carattere '|'. Per ogni film, crea un set contenente i suoi generi.
        Se la stringa dei generi è vuota, viene aggiunto un set vuoto.
        """
        self.genres_lists = []
        for genres_str in self.df["genres"]:
            if genres_str.strip() == "":
                self.genres_lists.append(set())  # Set vuoto se non ci sono generi
            else:
                self.genres_lists.append(set(genres_str.split("|")))  # Divide la stringa e crea un set di generi

    def _preprocess_all(self, vector_size: int, window: int, min_count: int) -> None:
        # Gestione dei valori mancanti per abstract e generi, riempiendoli con stringhe vuote
        self.df["dbpedia_abstract"] = self.df["dbpedia_abstract"].fillna("")
        self.df["genres"] = self.df["genres"].fillna("")

        # Preprocessa il testo degli abstract: tokenizzazione, rimozione stop words e punteggiatura
        self._preprocess_text()

        # Costruisce e allena il modello Word2Vec sugli abstract tokenizzati
        self.model = Word2Vec(sentences=self.tokenized_abstracts, vector_size=vector_size, window=window, min_count=min_count, workers=4)

        # Calcola i vettori documento (embeddings) per ogni abstract, mediando i vettori delle parole
        self._calculate_embeddings()

        # Prepara la lista dei generi: parsing della stringa dei generi in set di generi per ogni film
        self._parse_genres()

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """
        Calcola la similarità di Jaccard tra due set.
        La similarità di Jaccard è definita come |intersezione(set1, set2)| / |unione(set1, set2)|.
        """
        if not set1 and not set2:
            return 0.0  # Similarità 0 se entrambi i set sono vuoti
        intersection = len(set1 & set2)  # Cardinalità dell'intersezione
        union = len(set1 | set2)  # Cardinalità dell'unione
        return intersection / union if union > 0 else 0.0  # Jaccard similarity, evita divisione per zero

    def get_embeddings(self) -> np.ndarray:
        if self.embeddings is not None:
            return self.embeddings
        return []

    def get_idx(self, movie_title: str) -> int:
        # Ottieni l'indice del film dato nel DataFrame
        return self.df[self.df["title"] == movie_title].index[0]

    def _compute_genres_similarity(self, idx: int) -> np.ndarray:
        # Calcola la similarità basata sui generi (Jaccard similarity)
        # Confronta i generi del film dato con i generi di tutti gli altri film
        target_genres = self.genres_lists[idx]
        genre_sims = np.array([self._jaccard_similarity(target_genres, curr_genres_list) for curr_genres_list in self.genres_lists])
        return genre_sims

    def compute_similarity_scores(self, idx: int, alpha=0.90) -> np.ndarray:
        # 1. Calcola la similarità basata sull'abstract (cosine similarity), 
        #    confrontando l'embedding del film dato con gli embeddings di tutti gli altri film
        similarity_scores = cosine_similarity([self.embeddings[idx]], self.embeddings)
        abstract_sim = similarity_scores[0]

        # 2. Calcola la similarità basata sui generi (Jaccard similarity)
        genre_sims = self._compute_genres_similarity(idx)

        # 3. Normalizzazione delle similarità
        abstract_sim = (abstract_sim - abstract_sim.min()) / (abstract_sim.max() - abstract_sim.min())
        genre_sims = (genre_sims - genre_sims.min()) / (genre_sims.max() - genre_sims.min())

        #! 4. Combina le due similarità: abstract e generi usando il parametro alpha
        return alpha * abstract_sim + (1 - alpha) * genre_sims

    def recommend(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Raccomanda film simili ad un film dato.
        Utilizza una combinazione di similarità basata sull'abstract (cosine similarity sui document embeddings)
        e similarità basata sui generi (Jaccard similarity).
        """
        if movie_title not in self.df["title"].values:
            return []  # Restituisce lista vuota se il film non è nel dataset

        # Ottieni l'indice del film dato nel DataFrame
        movie_idx = self.get_idx(movie_title=movie_title)

        # Calcola la similarità basata sull'abstract e sul genres
        final_sim = self.compute_similarity_scores(idx=movie_idx)

        # Crea una lista di tuple (indice_film, punteggio_similarità)
        temp_sim_scores = list(enumerate(final_sim))
        temp_sim_scores = sorted(temp_sim_scores, key=lambda x: x[1], reverse=True)[: top_n + 1]  # Esclude il film stesso e prende i top_n
        sim_scores= []

        for i in temp_sim_scores:
            if i[0] != movie_idx:
                sim_scores += [i]
        # print(f"sim_scores: {[ (self.df.iloc[values[0]]['movieId'], values[1]) for values in sim_scores]}")

        # Estrai gli indici dei film raccomandati
        rec_indices = [i[0] for i in sim_scores]
        # print("movieId dei film consigliati: ", self.df.iloc[rec_indices]["movieId"].tolist()) # Stampa gli movieId per debugging/verifica

        # Restituisce il dataframe ordinato per scores dei film raccomandati
        return self.df.iloc[rec_indices]