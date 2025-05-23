# Modeling Musical Genre Trajectories through Pathlet Learning 


This repository provides Python code to reproduce experiments from the paper _"Modeling Musical Genre Trajectories through Pathlet Learning"_, accepted at the Proceedings of the Explainable User Models and Personalized Systems (ExUM) International Workshop on Transparent Personalization Methods based on Heterogeneous Personal Data, at The 33rd ACM Conference on User Modeling, Adaptation and Personalization (UMAP ’25), June 16th—19th 2025,New York, USA



## Abstract

The increasing availability of user data on music streaming platforms opens up new possibilities for analyzing music consumption. 
However, understanding the evolution of user preferences remains a complex challenge, particularly as their musical tastes change over time.
This paper uses the dictionary learning paradigm to model user trajectories across different musical genres. 
We define a new framework that captures recurring patterns in genre trajectories, called pathlets, enabling the creation of comprehensible trajectory embeddings. 
We show that pathlet learning reveals relevant listening patterns that can be analyzed both qualitatively and quantitatively. 
This work improves our understanding of users' interactions with music and opens up avenues of research into user behavior and fostering diversity in recommender systems.
A dataset of 2000 user histories tagged by genre over 17 months, supplied by a leading music streaming company, is also released with the code. 


## Dataset

https://zenodo.org/records/15341401

## Environment
```
networkx==3.3 
numpy==2.1.2 
pandas==2.2.3 
scikit-learn==1.5.2 
torch==2.4.1 
tqdm==4.66.5 
```

## Running the code

The repository must contain those folders (DEEZER being replacable by another dataset):  
```
├── data
│   └── DEEZER/histories.csv
├── processed_data/DEEZER/
└── results/DEEZER/
```

Processing and modeling dataset, evaluation :

```
python main.py
```
