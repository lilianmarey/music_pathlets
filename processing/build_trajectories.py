"""
This script samples trajectories for appearance and disappearance candidates 
using co-listening data.
"""

from processing.helpers import sample_trajectories
from helpers import import_pickle, save_pickle
from globals import *

########################################################################

X = import_pickle(f"{DEEZER_processed_path}X.pkl")

h_co_listening = import_pickle(f"{DEEZER_processed_path}h_co_listening.pkl")

all_tags = import_pickle(f"{DEEZER_processed_path}all_tags.pkl")

train_appearance_candidates = import_pickle(
    f"{DEEZER_processed_path}appearance_candidates_train.pkl"
)
train_disappearance_candidates = import_pickle(
    f"{DEEZER_processed_path}disappearance_candidates_train.pkl"
)
test_appearance_candidates = import_pickle(
    f"{DEEZER_processed_path}appearance_candidates_test.pkl"
)
test_disappearance_candidates = import_pickle(
    f"{DEEZER_processed_path}disappearance_candidates_test.pkl"
)

########################################################################

n_traj_per_candidate = 1000

train_appearance_trajectories = sample_trajectories(
    X=X,
    all_tags=all_tags,
    h_co_listening=h_co_listening,
    candidate_dict=train_appearance_candidates,
    n_traj_per_candidate=n_traj_per_candidate,
    max_period=DEEZER_K - 2,
)

train_disappearance_trajectories = sample_trajectories(
    X=X,
    all_tags=all_tags,
    h_co_listening=h_co_listening,
    candidate_dict=train_disappearance_candidates,
    n_traj_per_candidate=n_traj_per_candidate,
    max_period=DEEZER_K - 2,
)

test_appearance_trajectories = sample_trajectories(
    X=X,
    all_tags=all_tags,
    h_co_listening=h_co_listening,
    candidate_dict=test_appearance_candidates,
    n_traj_per_candidate=n_traj_per_candidate,
    max_period=DEEZER_K - 1,
)

test_disappearance_trajectories = sample_trajectories(
    X=X,
    all_tags=all_tags,
    h_co_listening=h_co_listening,
    candidate_dict=test_disappearance_candidates,
    n_traj_per_candidate=n_traj_per_candidate,
    max_period=DEEZER_K - 1,
)

###################################################################################

save_pickle(
    f"{DEEZER_processed_path}appearance_trajectories_train.pkl",
    train_appearance_trajectories,
)
save_pickle(
    f"{DEEZER_processed_path}disappearance_trajectories_train.pkl",
    train_disappearance_trajectories,
)

save_pickle(
    f"{DEEZER_processed_path}appearance_trajectories_test.pkl",
    test_appearance_trajectories,
)
save_pickle(
    f"{DEEZER_processed_path}disappearance_trajectories_test.pkl",
    test_disappearance_trajectories,
)
