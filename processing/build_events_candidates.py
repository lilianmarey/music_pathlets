"""
Identifies appearance and disappearance candidates from user-tag interactions 
using pre-computed allocations.
"""

from processing.helpers import compute_candidates
from helpers import import_pickle, save_pickle
from globals import *

########################################################################

X = import_pickle(f"{XXXX_processed_path}X.pkl")
Y = import_pickle(f"{XXXX_processed_path}Y.pkl")
all_users = import_pickle(f"{XXXX_processed_path}all_users.pkl")
all_tags = import_pickle(f"{XXXX_processed_path}all_tags.pkl")

########################################################################

train_appearance_candidates, train_disappearance_candidates = compute_candidates(
    X, Y, all_users, all_tags, shift=1, K=XXXX_K
)
test_appearance_candidates, test_disappearance_candidates = compute_candidates(
    X, Y, all_users, all_tags, shift=0, K=XXXX_K
)

########################################################################

save_pickle(
    f"{XXXX_processed_path}appearance_candidates_train.pkl",
    train_appearance_candidates,
)
save_pickle(
    f"{XXXX_processed_path}disappearance_candidates_train.pkl",
    train_disappearance_candidates,
)

save_pickle(
    f"{XXXX_processed_path}appearance_candidates_test.pkl",
    test_appearance_candidates,
)
save_pickle(
    f"{XXXX_processed_path}disappearance_candidates_test.pkl",
    test_disappearance_candidates,
)
