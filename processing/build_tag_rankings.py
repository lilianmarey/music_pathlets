"""
This script computes user-tag rankings based on interaction values from the allocations data.
"""

from tqdm import tqdm
from helpers import import_pickle, save_pickle
from globals import *

########################################################################

X = import_pickle(f"{XXXX_processed_path}X.pkl")
all_users = import_pickle(f"{XXXX_processed_path}all_users.pkl")
all_tags = import_pickle(f"{XXXX_processed_path}all_tags.pkl")

########################################################################

ranking = dict()
for user_id in tqdm(all_users):
    tag_scores = dict([(tag, 0) for tag in all_tags])
    for tag in all_tags:
        for t in range(XXXX_K - 1):
            tag_scores[tag] += X[(user_id, t, tag)]
    ranking[user_id] = [
        tag for tag, _ in sorted(list(tag_scores.items()), key=lambda x: -x[1])
    ]

########################################################################

save_pickle(f"{XXXX_processed_path}users_tag_ranking.pkl", ranking)
