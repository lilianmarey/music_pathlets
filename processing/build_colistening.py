"""
Processes user listening history to compute co-listening occurrences 
between tags over defined time periods.
"""

import pandas as pd
from tqdm import tqdm
from helpers import increment_or_add, import_pickle, save_pickle
from globals import *

########################################################################

df = pd.read_csv(f"{XXXX_processed_path}cut_histories.pkl")
X = import_pickle(f"{XXXX_processed_path}X.pkl")
all_users = import_pickle(f"{XXXX_processed_path}all_users.pkl")
all_tags = import_pickle(f"{XXXX_processed_path}all_tags.pkl")

########################################################################

h_co_listening = dict()

for user_id in tqdm(all_users):
    df_user = df[df["user_id"] == user_id]
    user_listening_list = [tuple(i) for i in df_user[["ts", "tag"]].to_numpy()]

    for t in range(XXXX_K):
        t_user_listening_list = [
            stream[1]
            for stream in user_listening_list
            if stream[0] >= XXXX_time_periods[t][0]
            and stream[0] < XXXX_time_periods[t][1]
        ]
        if len(t_user_listening_list) > 1:
            increment_or_add(
                h_co_listening,
                (user_id, t, t_user_listening_list[0], t_user_listening_list[1]),
            )
            increment_or_add(
                h_co_listening,
                (user_id, t, t_user_listening_list[-1], t_user_listening_list[-2]),
            )
            for i in range(1, len(t_user_listening_list) - 1):
                increment_or_add(
                    h_co_listening,
                    (
                        user_id,
                        t,
                        t_user_listening_list[i],
                        t_user_listening_list[i - 1],
                    ),
                )
                increment_or_add(
                    h_co_listening,
                    (
                        user_id,
                        t,
                        t_user_listening_list[i],
                        t_user_listening_list[i + 1],
                    ),
                )

########################################################################

save_pickle(f"{XXXX_processed_path}h_co_listening.pkl", h_co_listening)
