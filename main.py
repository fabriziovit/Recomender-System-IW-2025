import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF, XSD
import random

# Load MovieLens dataset
def load_movielens_data(path):
    movies = pd.read_csv(path + 'movies.csv')
    ratings = pd.read_csv(path + 'ratings.csv')
    tags = pd.read_csv(path + 'tags.csv')
    return movies, ratings, tags

# Query DBpedia for movie information
def query_dbpedia(movie_title):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
    SELECT ?abstract ?director ?actor
    WHERE {{
      ?movie rdf:type dbo:Film ;
             rdfs:label "{movie_title}"@en ;
             dbo:abstract ?abstract ;
             dbo:director ?director ;
             dbo:starring ?actor .
      FILTER (lang(?abstract) = 'en')
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

# Knowledge-based recommender system
class KnowledgeBasedRecommender:
    def __init__(self, movies, dbpedia_info):
        self.movies = movies
        self.dbpedia_info = dbpedia_info
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
    def fit(self):
        movie_features = self.movies['genres'] + ' ' + self.dbpedia_info['abstract']
        self.tfidf_matrix = self.tfidf.fit_transform(movie_features)
        
    def recommend(self, movie_id, n=5):
        movie_idx = self.movies.index[self.movies['movieId'] == movie_id][0]
        cosine_sim = cosine_similarity(self.tfidf_matrix[movie_idx], self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices]

# Multi-Armed Bandit implementation
class EpsilonGreedyMAB:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)
        
    def select_arm(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.q_values)
        
    def update(self, arm, reward):
        self.n_pulls[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.n_pulls[arm]

# Experimental verification
def evaluate_recommender(recommender, test_set, mab):
    precision = []
    for movie_id in test_set['movieId']:
        recommended = recommender.recommend(movie_id)
        arm = mab.select_arm()
        selected_movie = recommended.iloc[arm]
        actual_rating = test_set[test_set['movieId'] == selected_movie['movieId']]['rating'].values[0]
        precision.append(1 if actual_rating >= 4 else 0)
        mab.update(arm, actual_rating / 5)  # Normalize reward
    return np.mean(precision)

# Main execution
if __name__ == "__main__":
    # Load data
    movies, ratings, tags = load_movielens_data('path/to/movielens/data/')
    
    # Query DBpedia for each movie (this might take a while)
    dbpedia_info = {}
    for title in movies['title']:
        dbpedia_info[title] = query_dbpedia(title)
    
    # Create and fit recommender
    recommender = KnowledgeBasedRecommender(movies, dbpedia_info)
    recommender.fit()
    
    # Create MAB
    mab = EpsilonGreedyMAB(5)  # 5 arms for top 5 recommendations
    
    # Split data for evaluation
    train_set = ratings.sample(frac=0.8, random_state=42)
    test_set = ratings.drop(train_set.index)
    
    # Evaluate
    precision = evaluate_recommender(recommender, test_set, mab)
    print(f"Precision: {precision}")