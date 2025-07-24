import logging
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ContentBasedRecommender:
    """
    Class for a content-based recommendation system.
    This system uses movie abstracts and genres to provide recommendations.
    It leverages Word2Vec for semantic representation of abstracts and cosine similarity
    to compare abstracts. For genres, it uses Jaccard similarity.
    The abstract and genre similarities are combined to generate a final score.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
    ):
        """Initialize the ContentBasedRecommender."""
        self.df = df

        self._preprocess_all(vector_size=vector_size, window=window, min_count=min_count)

        logging.info(f"Word2Vec vocabulary created with {len(self.model.wv.index_to_key)} unique words.")

    def _preprocess_text(self) -> None:
        """Preprocess the text of abstracts using Gensim.
        Performs the following operations:
        1. Tokenization and normalization: uses simple_preprocess (lowercase, removes punctuation and tokenizes).
        2. Removal of Gensim's predefined stopwords.
        """
        self.tokenized_abstracts = []
        # List of lists, where each inner list contains the tokens of a certain abstract
        self.tokenized_abstracts = [[word for word in simple_preprocess(text) if word not in STOPWORDS] for text in self.df["dbpedia_abstract"]]

    def _calculate_embeddings(self) -> None:
        """Calculate embeddings for each abstract as the average of Word2Vec vectors."""
        embeddings = []
        for curr_tokens in self.tokenized_abstracts:
            word_vectors = [self.model.wv[word] for word in curr_tokens if word in self.model.wv]  # Retrieve embeddings for each word in tokens

            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
                assert doc_vector.shape[0] == self.model.vector_size, f"Error: dimension {doc_vector.shape[0]} instead of {self.model.vector_size}"
            else:
                doc_vector = np.zeros(self.model.vector_size)
            embeddings.append(doc_vector)
        logging.info("End of _calculate_embeddings")
        self.embeddings = np.array(embeddings)

    def _parse_genres(self) -> None:
        """Parse the genres column transforming strings into sets of genres.
        Genres are separated by the '|' character. For each movie, creates a set containing its genres.
        If the genre string is empty, an empty set is added.
        """
        self.genres_lists = []
        for genres_str in self.df["genres"]:
            if genres_str.strip() == "":
                self.genres_lists.append(set())  # Empty set if there are no genres
            else:
                self.genres_lists.append(set(genres_str.split("|")))  # Split the string and create a set of genres

    def _preprocess_all(self, vector_size: int, window: int, min_count: int) -> None:
        # Handle missing values for abstracts and genres, filling them with empty strings
        self.df["dbpedia_abstract"] = self.df["dbpedia_abstract"].fillna("")
        self.df["genres"] = self.df["genres"].fillna("")

        # Preprocess abstract text: tokenization, stopword removal and punctuation
        self._preprocess_text()

        # Build and train the Word2Vec model on tokenized abstracts
        self.model = Word2Vec(sentences=self.tokenized_abstracts, vector_size=vector_size, window=window, min_count=min_count, workers=4)

        # Calculate document vectors (embeddings) for each abstract, averaging word vectors
        self._calculate_embeddings()

        # Prepare the genre list: parsing genre strings into sets of genres for each movie
        self._parse_genres()

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets.
        Jaccard similarity is defined as |intersection(set1, set2)| / |union(set1, set2)|.
        """
        if not set1 and not set2:
            return 0.0  # Similarity 0 if both sets are empty
        intersection = len(set1 & set2)  # Cardinality of intersection
        union = len(set1 | set2)  # Cardinality of union
        return intersection / union if union > 0 else 0.0  # Jaccard similarity, avoid division by zero

    def get_embeddings(self) -> np.ndarray:
        if self.embeddings is not None:
            return self.embeddings
        return []

    def get_idx(self, movie_title: str) -> int:
        """Get the index of the given movie in the DataFrame"""
        return self.df[self.df["title"] == movie_title].index[0]

    def _compute_genres_similarity(self, idx: int) -> np.ndarray:
        """Calculate genre-based similarity (Jaccard similarity)
        Compare the genres of the given movie with the genres of all other movies"""
        target_genres = self.genres_lists[idx]
        genre_sims = np.array([self._jaccard_similarity(target_genres, curr_genres_list) for curr_genres_list in self.genres_lists])
        return genre_sims

    def compute_similarity_scores(self, idx: int, alpha=0.90) -> np.ndarray:
        """Calculate abstract-based similarity (cosine similarity)
        comparing the embedding of the given movie with the embeddings of all other movies"""

        similarity_scores = cosine_similarity([self.embeddings[idx]], self.embeddings)
        abstract_sim = similarity_scores[0]

        # Calculate genre-based similarity (Jaccard similarity)
        genre_sims = self._compute_genres_similarity(idx)

        # Normalize similarities
        abstract_sim = (abstract_sim - abstract_sim.min()) / (abstract_sim.max() - abstract_sim.min())
        genre_sims = (genre_sims - genre_sims.min()) / (genre_sims.max() - genre_sims.min())

        #! 4. Combine the two similarities: abstract and genres using the alpha parameter
        return alpha * abstract_sim + (1 - alpha) * genre_sims

    def recommend(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        """Recommend movies similar to a given movie.
        Uses a combination of abstract-based similarity (cosine similarity on document embeddings)
        and genre-based similarity (Jaccard similarity).
        """
        if movie_title not in self.df["title"].values:
            return []  # Return empty list if the movie is not in the dataset

        # Get the index of the given movie in the DataFrame
        movie_idx = self.get_idx(movie_title=movie_title)

        # Calculate similarity based on abstract and genres
        final_sim = self.compute_similarity_scores(idx=movie_idx)

        # Create a list of tuples (movie_index, similarity_score)
        temp_sim_scores = list(enumerate(final_sim))
        temp_sim_scores = sorted(temp_sim_scores, key=lambda x: x[1], reverse=True)[: top_n + 1]  # Exclude the movie itself and take top_n
        sim_scores = []

        for i in temp_sim_scores:
            if i[0] != movie_idx:
                sim_scores += [i]
        logging.info(f"sim_scores: {[ (self.df.iloc[values[0]]['movieId'], values[1]) for values in sim_scores]}")

        # Extract indices of recommended movies
        rec_indices = [i[0] for i in sim_scores]
        logging.info("movieId of recommended movies: ", self.df.iloc[rec_indices]["movieId"].tolist())  # Print movieIds for debugging/verification

        # Return the dataframe ordered by scores of recommended movies
        return self.df.iloc[rec_indices]
