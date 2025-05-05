"""
Processes appearance and disappearance trajectories to compute 
subpath counts for candidates.
"""

from modeling.helpers import pool_process_candidates
from helpers import import_pickle, save_pickle
from globals import *

########################################################################

all_users = import_pickle(f"{XXXX_processed_path}all_users.pkl")
appearance_trajectories = import_pickle(
    f"{XXXX_processed_path}appearance_trajectories_train.pkl"
)
disappearance_trajectories = import_pickle(
    f"{XXXX_processed_path}disappearance_trajectories_train.pkl"
)
users_tag_ranking = import_pickle(f"{XXXX_processed_path}users_tag_ranking.pkl")

########################################################################

appearance_candidates = list(appearance_trajectories.keys())
disappearance_candidates = list(disappearance_trajectories.keys())

lengths = range(2, 11)
n_traj_per_candidate = 1000
traj_dict = {
    "appearance": appearance_trajectories,
    "disappearance": disappearance_trajectories,
}

appearance_counter = pool_process_candidates(
    traj_dict,
    users_tag_ranking,
    n_traj_per_candidate,
    lengths,
    appearance_candidates,
    "appearance",
).most_common(int(10e6))

disappearance_counter = pool_process_candidates(
    traj_dict,
    users_tag_ranking,
    n_traj_per_candidate,
    lengths,
    disappearance_candidates,
    "disappearance",
).most_common(int(10e6))

########################################################################

save_pickle(
    f"{XXXX_processed_path}appearance_subpaths_counter.pkl",
    appearance_counter,
)
save_pickle(
    f"{XXXX_processed_path}disappearance_subpaths_counter.pkl",
    disappearance_counter,
)
