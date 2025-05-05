"""
Computes embeddings for appearance and disappearance trajectories 
using pre-trained dictionaries.
"""

from helpers import import_pickle
from modeling.helpers import compute_embeddings
from globals import *

####################################################################################

appearance_D = import_pickle(f"{XXXX_processed_path}D_appearance.pkl")
disappearance_D = import_pickle(f"{XXXX_processed_path}D_disappearance.pkl")

users_tag_ranking = import_pickle(f"{XXXX_processed_path}users_tag_ranking.pkl")
train_appearance_trajectories = import_pickle(
    f"{XXXX_processed_path}appearance_trajectories_train.pkl"
)
train_disappearance_trajectories = import_pickle(
    f"{XXXX_processed_path}disappearance_trajectories_train.pkl"
)
test_appearance_trajectories = import_pickle(
    f"{XXXX_processed_path}appearance_trajectories_test.pkl"
)
test_disappearance_trajectories = import_pickle(
    f"{XXXX_processed_path}disappearance_trajectories_test.pkl"
)

####################################################################################

compute_embeddings(
    appearance_D,
    users_tag_ranking,
    train_appearance_trajectories,
    f"{XXXX_processed_path}appearance_embeddings_train.csv",
)
compute_embeddings(
    disappearance_D,
    users_tag_ranking,
    train_disappearance_trajectories,
    f"{XXXX_processed_path}disappearance_embeddings_train.csv",
)
compute_embeddings(
    appearance_D,
    users_tag_ranking,
    test_appearance_trajectories,
    f"{XXXX_processed_path}appearance_embeddings_test.csv",
)
compute_embeddings(
    disappearance_D,
    users_tag_ranking,
    test_disappearance_trajectories,
    f"{XXXX_processed_path}disappearance_embeddings_test.csv",
)
