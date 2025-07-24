# 🎬 Movie Recommendation System

**Course Project – Intelligent Web**

**Authors:** Giuseppe Balzano, Fabrizio Vitale

---

## 1. Introduction

### 1.1 Project Description

This project implements a modular and extensible movie recommendation system developed for academic research and experimentation. It uses the **MovieLens Small (ml-latest-small)** dataset and enriches it with external semantic data from **DBpedia** and **Wikidata** via **SPARQL** queries. The system integrates multiple recommendation strategies, including **content-based filtering**, **collaborative filtering**, **matrix factorization**, and **multi-armed bandit (MAB)** models.

### 1.2 Core Features

* ✅ **Hybrid Recommendation Engine** (content-based, collaborative, knowledge-based, MAB)
* 🧠 **Semantic Enrichment** through SPARQL queries to DBpedia/Wikidata
* ⚙️ **Modular Python Implementation** (Flask API + Web Frontend)
* 📊 **Evaluation Metrics**: MAE and RMSE

### 1.3 Target Use Cases

* Personalized content recommendation
* Evaluation and benchmarking of hybrid recommender algorithms
* Cold-start user handling via semantic and bandit-based methods

---

## 2. System Architecture

### 2.1 Backend (Python + Flask)

* RESTful API for recommendation and search queries
* Integration with external knowledge sources (SPARQL)
* Core logic for predictions and model orchestration

### 2.2 Frontend (HTML/CSS/JS)

* Interactive web-based UI
* Real-time movie search and personalized recommendation display
* Frontend-backend communication via API endpoints

### 2.3 Design Principles

* Clear modular separation (data, logic, UI)
* Easy extensibility for new recommendation strategies
* Support for real-time interaction and explainable outputs

---

## 3. User Interface Overview

| Module                   | Functionality                                            |
| ------------------------ | -------------------------------------------------------- |
| **Search & Info**        | Title search, full movie data, user rating system        |
| **DBpedia Recs**         | Director-based recommendations via SPARQL                |
| **Content-Based Recs**   | Abstract similarity + genre matching (Word2Vec, Jaccard) |
| **Item-Based CF**        | Recommendations based on similar movies                  |
| **User-Based CF**        | Suggestions based on similar users                       |
| **Matrix Factorization** | Latent factor predictions (SGD)                          |
| **MAB**                  | Exploration/exploitation via ε-greedy                    |
| **Rating Prediction**    | Rating estimates using KNN + SGD                         |

---

## 4. Installation & Usage

### 4.1 Setup Instructions

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
python app.py
```

Access the app at: `http://localhost:5000`

### 4.2 Requirements

* Python ≥ 3.7
* Flask
* scikit-learn
* pandas, numpy
* gensim
* SPARQLWrapper
* Internet connection (required for SPARQL queries)

---

## 5. Content-Based Filtering

### 5.1 Director-Based (DBpedia/Wikidata)

* Uses SPARQL queries to retrieve recent movies by the same director.
* Results sorted by release year and limited to top 5 titles.

### 5.2 Abstract + Genre Similarity

* Abstracts vectorized using **Word2Vec**
* Genre similarity computed with **Jaccard index**

**Scoring formula:**

```python
score = α * abstract_sim + (1 - α) * genre_sim  # α = 0.90
```

**Explanation:**
This computes a weighted combination of semantic (abstract) and categorical (genre) similarity. Abstracts have higher priority (`α = 0.90`), emphasizing semantic relevance.

---

## 6. Collaborative Filtering

### 6.1 Item-Based CF

* Computes **cosine similarity** between item vectors.
* Uses KNN to find most similar movies to those rated by the user.

### 6.2 User-Based CF

* Computes **Pearson correlation** between users.
* Predicts ratings using mean-centered collaborative filtering:

```math
r̂_uj = [Σ sim(u,v)·(r_vj - μ_v) / Σ |sim(u,v)|] + μ_u
```

**Explanation:**
Predicts the rating of user `u` for item `j` by aggregating weighted deviations of similar users’ ratings, normalized by user averages.

### 6.3 Matrix Factorization (SGD)

* Based on **Koren et al. (2009)**
* Prediction formula:

```math
r̂_uj = μ + b_u + b_j + Σ_k x_uk · y_kj
```

**Explanation:**
Models latent features for users and items, plus global and local biases. Trained using stochastic gradient descent with regularization.

---

## 7. Multi-Armed Bandit (MAB)

### 7.1 ε-Greedy Strategy

* Exploits best-known recommendation with `1−ε` probability
* Explores other options with `ε` probability

**Reward function:**

```math
reward = predicted_rating / (1 + log(1 + selections))
```

**Explanation:**
Normalizes predicted rating based on the number of times an item has been selected, favoring novel and high-quality recommendations.

### 7.2 ε Decay Strategies

| Strategy    | Formula                          |
| ----------- | -------------------------------- |
| Linear      | ε(t) = ε₀ − d × t                |
| Logarithmic | ε(t) = ε₀ / (1 + λ · log(t + 1)) |
| Exponential | ε(t) = ε₀ × λᵗ                   |

**Explanation:**
Gradual decay of `ε` helps the model shift from exploration to exploitation over time.

---

## 8. Dataset & Knowledge Integration

### 8.1 MovieLens Small

* Dataset: **ml-latest-small**
* \~100,000 ratings, \~9,000 movies, 600 users
* Includes metadata: genres, titles, timestamps

### 8.2 Data Splitting

* 70% training
* 15% validation
* 15% test
* Stratified sampling with sparsity preservation

### 8.3 Knowledge Enrichment

* **DBpedia**: Abstracts, directors
* **Wikidata**: Release years
* SPARQL queries executed with fallback using `COALESCE`

---

## 9. Evaluation Metrics

### 9.1 Mean Absolute Error (MAE)

```math
MAE = (1 / |T|) Σ |r̂_ui - r_ui|
```

**Explanation:**
Measures the average absolute deviation between predicted and true ratings. Lower values indicate better performance.

### 9.2 Root Mean Squared Error (RMSE)

```math
RMSE = sqrt[(1 / |T|) Σ (r̂_ui - r_ui)^2]
```

**Explanation:**
Gives more weight to larger errors; useful for penalizing large deviations.

---

## 10. Experimental Results

### 10.1 KNN (User/Item-Based)

* **Best MAE**: 0.6744 (600 neighbors, normalized)
* **Best RMSE**: 0.8868
* **Configuration**: Mean-centering, cosine similarity

### 10.2 Matrix Factorization (SGD)

* **Best MAE**: 0.6577 (200 latent factors, λ=0)
* **Best RMSE**: 0.8568 (400 latent factors, λ=0)
* Outperformed all KNN-based approaches

### 10.3 Multi-Armed Bandit (ε-Greedy)

| Decay Strategy | Cumulative Reward | Exploitation Rate   |
| -------------- | ----------------- | ------------------- |
| Exponential    | 25.5M             | Best balance        |
| Logarithmic    | 25.2M             | Moderate            |
| Linear         | 23.1M             | Highest exploration |

### 10.4 Comparison with Literature

| Model               | MAE   | RMSE  |
| ------------------- | ----- | ----- |
| User Mean           | 0.839 | —     |
| Slope One           | 0.737 | —     |
| SVD++ (Surprise)    | 0.721 | 0.919 |
| **KNN (this work)** | **0.674** | **0.887** |
| **MF (this work)**  | **0.662** | **0.863** |

---

## 11. License

This project is intended for educational and research use.
A suitable open-source license (e.g., MIT or Apache 2.0) can be added based on use cases.

---

## 12. Acknowledgments

Thanks to the teams behind **MovieLens**, **DBpedia**, and **Wikidata** for providing high-quality open datasets that enabled this project.
