import random
import datetime
import numpy as np
from tqdm import tqdm


def datetime_to_timestamp(day, month, year):
    """
    Convert a date into a Unix timestamp.
    """
    return int(datetime.date(year=year, month=month, day=day).strftime("%s"))


def timestamp_to_time_period(ts, time_periods):
    """
    Maps a timestamp to a time period based on defined ranges.
    """
    for t, (ts_min, ts_max) in time_periods.items():
        if ts >= ts_min and ts < ts_max:
            return t
    return "error"


def compute_allocations(df, all_users, all_tags, time_periods, stream_threshold):
    """
    Computes allocation of tags for each user across time periods based on stream counts.
    """
    X = dict()
    X_counts = dict()
    not_valid_users = set()

    for user_id in tqdm(all_users):
        df_user = df[df["user_id"] == user_id]
        for t in time_periods.keys():
            df_period = df_user[df_user["time_period"] == t]
            allocation_dict = dict(
                df_period["tag"].value_counts().reset_index().to_numpy()
            )
            x_count = [
                (
                    allocation_dict[all_tags[i]]
                    if (
                        all_tags[i] in allocation_dict.keys()
                        and allocation_dict[all_tags[i]] >= stream_threshold
                    )
                    else 0
                )
                for i in range(len(all_tags))
            ]
            if np.sum(x_count) == 0:
                not_valid_users.add(user_id)
            else:
                x_count = np.array(x_count)
                x = list(x_count / np.sum(x_count))
                for tag_index, _ in enumerate(x):
                    X[(user_id, t, all_tags[tag_index])] = x[tag_index]
                    X_counts[(user_id, t, all_tags[tag_index])] = x_count[tag_index]

        for user_id in not_valid_users:
            for t in time_periods:
                for tag in all_tags:
                    if (user_id, t, tag) in X.keys():
                        del X[(user_id, t, tag)]

    valid_users = list(set([user_id for (user_id, _, _) in X.keys()]))

    return X, X_counts, valid_users


def delete_non_listen_genres(df, X):
    """
    Removes rows from the DataFrame where the user has no valid streams for a genre.
    """
    df["user_id_t_tag"] = list(
        zip(df.user_id.tolist(), df.time_period.tolist(), df.tag.tolist())
    )
    kept_streams = [key for key, val in X.items() if val != 0]
    df = df[df["user_id_t_tag"].isin(kept_streams)]
    df = df.drop(columns=["user_id_t_tag"])
    return df


def compute_candidates(X, Y, all_users, all_tags, shift, K):
    """
    Identifies appearance and disappearance candidates for tags based on past behavior.
    """
    appearance_candidates = dict()
    disappearance_candidates = dict()

    for user_id in tqdm(all_users):
        for tag in all_tags:
            if (
                X[(user_id, K - 1 - shift, tag)] == 0
                and np.sum(
                    [X[(user_id, t, tag)] for t in range(K - 3 - shift, K - shift)]
                )
                > 0
            ):
                appearance_candidates[(user_id, tag)] = int(Y[user_id, tag] > 0)
            elif X[(user_id, K - 1 - shift, tag)] > 0:
                disappearance_candidates[(user_id, tag)] = int(Y[user_id, tag] == 0)
            else:
                pass

    return appearance_candidates, disappearance_candidates


def sample_trajectories(
    X, all_tags, h_co_listening, candidate_dict, n_traj_per_candidate, max_period
):
    """
    Samples trajectories of tag co-listening behavior for candidates across time periods.
    """

    trajectories = dict(((user_id, tag), []) for user_id, tag in candidate_dict.keys())

    for user_id, tag in tqdm(candidate_dict.keys()):
        trajectory = []
        for t in range(max_period + 1):
            cl_tags = []
            for tag_cl in all_tags:
                try:
                    cl_tags.append((tag_cl, h_co_listening[(user_id, t, tag, tag_cl)]))
                except KeyError:
                    pass
            if cl_tags == []:
                for tag_cl in all_tags:
                    cl_tags.append((tag_cl, X[(user_id, t, tag_cl)]))
            tag_names = [cl_tag[0] for cl_tag in cl_tags]
            tag_weight = [cl_tag[1] for cl_tag in cl_tags]
            traj_step = random.choices(
                tag_names, weights=tag_weight, k=n_traj_per_candidate
            )
            trajectory.append(traj_step)
        for i in range(n_traj_per_candidate):
            trajectories[(user_id, tag)].append(
                tuple([trajectory[t][i] for t in range(max_period + 1)])
            )
    return trajectories
