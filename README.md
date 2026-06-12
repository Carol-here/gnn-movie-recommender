# Graph Neural Network Movie Recommendation System

## Overview

This project implements an end-to-end movie recommendation system using Graph Neural Networks (GraphSAGE) on the MovieLens 100K dataset.

Unlike traditional recommendation approaches that operate on user-item matrices, this system models the recommendation problem as a graph where users and movies are represented as nodes and user ratings are represented as edges. By leveraging graph structure, the model learns latent representations of both users and movies and generates personalized recommendations through link prediction.

The project also includes an interactive Streamlit application that allows users to select two movies and receive ten personalized recommendations along with dynamically generated explanations based on genre overlap and recommendation scores.

---

## Why Graph Neural Networks?

Recommendation systems naturally form graphs.

Traditional collaborative filtering methods such as SVD treat interactions as entries in a matrix:

User → Movie → Rating

While effective, these approaches do not explicitly model relationships between users and movies as connected entities.

Graph Neural Networks enable:

* Message passing between connected users and movies
* Learning from graph neighborhood information
* Representation learning without extensive feature engineering
* Flexible integration of additional content features such as genres

For recommendation tasks, this allows the model to learn not only which movies a user interacted with, but also how those interactions relate to the broader network of users and movies.

---

## Dataset

MovieLens 100K

Dataset Statistics:

| Metric      | Value   |
| ----------- | ------- |
| Ratings     | 100,000 |
| Users       | 943     |
| Movies      | 1,682   |
| Total Nodes | 2,625   |
| Graph Edges | 200,000 |

The graph was constructed as a bipartite network:

* User nodes: 943
* Movie nodes: 1,682
* User-Movie interactions: 100,000
* Bidirectional edges: 200,000

---

## Feature Engineering

### User Features

For each user:

* Average rating
* Number of ratings
* Rating standard deviation

### Movie Features

Movie genre vectors extracted from MovieLens metadata.

Genres include:

* Action
* Adventure
* Animation
* Children's
* Comedy
* Crime
* Documentary
* Drama
* Fantasy
* Film-Noir
* Horror
* Musical
* Mystery
* Romance
* Sci-Fi
* Thriller
* War
* Western

These features were incorporated into node representations before GraphSAGE training.

---

## Model Architecture

Graph Neural Network:

GraphSAGE

Architecture:

Input Features
↓
GraphSAGE Layer
↓
ReLU
↓
GraphSAGE Layer
↓
64-Dimensional Node Embeddings
↓
Link Prediction Decoder

Embedding Size:

64 Dimensions

Optimizer:

Adam

Learning Rate:

0.01

Training Epochs:

50

---

## Results

### GraphSAGE Performance

| Metric         | Value |
| -------------- | ----- |
| Validation AUC | 0.799 |
| Test AUC       | 0.803 |

The model successfully learned meaningful user and movie representations and achieved strong link prediction performance on unseen interactions.

### Collaborative Filtering Baseline

SVD Benchmark:

| Metric       | Value |
| ------------ | ----- |
| RMSE         | 0.936 |
| Precision@10 | 0.586 |

This baseline was used to compare traditional recommendation approaches with graph-based representation learning.

---

## Recommendation Pipeline

1. User selects two movies.
2. Embeddings for both movies are retrieved.
3. Movie embeddings are averaged to create a preference profile.
4. Cosine similarity is computed against all candidate movie embeddings.
5. Top-ranked movies are returned.
6. Dynamic explanations are generated using genre overlap statistics.

---

## Streamlit Application

Features:

* Interactive movie selection
* Top-10 personalized recommendations
* Graph-based recommendation engine
* Dynamic recommendation explanations
* Genre overlap analysis
* Similarity-based ranking

Example Explanation:

"67% genre overlap with your selected movies."

"Users who enjoy Action and Sci-Fi content frequently interact with similar titles."

---

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* GraphSAGE
* Pandas
* NumPy
* Scikit-Learn
* Streamlit

---

## Project Structure

```text
gnn-movie-recommender/

├── train.ipynb
├── app.py
├── requirements.txt
├── README.md

├── models/
│   └── graphsage.pt

├── artifacts/
│   ├── movie_mapping.pkl
│   ├── movie_titles.pkl
│   ├── movie_genres.pkl
│   ├── node_features.pt
│   └── edge_index.pt

└── screenshots/
```

---

## Future Improvements

* MLflow experiment tracking
* Docker deployment
* FastAPI inference service
* Cold-start user evaluation
* Graph Attention Networks (GAT)
* Hybrid recommendation models
* Distributed graph training

---

## Milestones

* Constructed a graph containing 2,625 nodes and 200,000 interactions.
* Learned 64-dimensional graph embeddings using GraphSAGE.
* Achieved a Test AUC of 0.803 on the MovieLens 100K dataset.
* Integrated user statistics and movie genre features into node representations.
* Built a complete recommendation application with an interactive Streamlit frontend.
* Demonstrated how graph-based learning can be applied to recommendation systems beyond traditional collaborative filtering methods.
