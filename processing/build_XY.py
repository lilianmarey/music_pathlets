"""
Pprocesses historical user data to compute allocations and clean the dataset. 
"""

import pandas as pd
from tqdm import tqdm

from processing.helpers import (
    compute_allocations,
    delete_non_listen_genres,
    timestamp_to_time_period,
)
from helpers import save_pickle
from globals import *

tqdm.pandas()

########################################################################

# Import raw data
df = pd.read_csv("data/DEEZER/histories.csv")

# Compute time periods
df["time_period"] = df["ts"].progress_apply(
    lambda ts: timestamp_to_time_period(ts, DEEZER_time_periods)
)

# Define user sets
all_users = df.user_id.unique().tolist()
all_tags = df.tag.unique().tolist()

# Build X
X, X_counts, valid_users = compute_allocations(
    df, all_users, all_tags, DEEZER_time_periods, DEEZER_stream_threshold
)
print(f"Number of valid users : {len(valid_users)}")

# Build Y
Y = dict()

for user_id in valid_users:
    for tag in all_tags:
        Y[(user_id, tag)] = X[(user_id, DEEZER_K, tag)]
        del X[(user_id, DEEZER_K, tag)]

# Clean df
df = delete_non_listen_genres(df, X)

# Save data
df.to_csv(f"{DEEZER_processed_path}cut_histories.pkl", index=False)
save_pickle(f"{DEEZER_processed_path}all_users.pkl", valid_users)
save_pickle(f"{DEEZER_processed_path}all_tags.pkl", all_tags)
save_pickle(f"{DEEZER_processed_path}X.pkl", X)
save_pickle(f"{DEEZER_processed_path}X_counts.pkl", X_counts)
save_pickle(f"{DEEZER_processed_path}Y.pkl", Y)
