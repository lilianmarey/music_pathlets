"""
Computes dictionaries for appearance and disappearance trajectories 
using pathlet learning.
"""

from modeling.helpers import compute_dictionary
from helpers import import_pickle, save_pickle
from globals import *

########################################################################

all_users = import_pickle(f"{XXXX_processed_path}all_users.pkl")
users_tag_ranking = import_pickle(f"{XXXX_processed_path}users_tag_ranking.pkl")
appearance_trajectories = import_pickle(
    f"{XXXX_processed_path}appearance_trajectories_train.pkl"
)
disappearance_trajectories = import_pickle(
    f"{XXXX_processed_path}disappearance_trajectories_train.pkl"
)
appearance_sub_paths_counter = import_pickle(
    f"{XXXX_processed_path}appearance_subpaths_counter.pkl"
)
disappearance_sub_paths_counter = import_pickle(
    f"{XXXX_processed_path}disappearance_subpaths_counter.pkl"
)

###################################################################

n_traj_for_pathlet_learning = 5000
n_candidates = 10000

learning_parameters = {
    "lambda_": 0.0025,
    "n_steps": 5000,
    "dictionary_size": 1000,
}

###################################################################

appearance_D = compute_dictionary(
    appearance_trajectories,
    appearance_sub_paths_counter,
    n_traj_for_pathlet_learning,
    n_candidates,
    learning_parameters,
    users_tag_ranking,
)

disappearance_D = compute_dictionary(
    disappearance_trajectories,
    disappearance_sub_paths_counter,
    n_traj_for_pathlet_learning,
    n_candidates,
    learning_parameters,
    users_tag_ranking,
)

###################################################################

save_pickle(f"{XXXX_processed_path}D_appearance.pkl", appearance_D)
save_pickle(f"{XXXX_processed_path}D_disappearance.pkl", disappearance_D)
