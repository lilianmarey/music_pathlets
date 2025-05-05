import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter
from modeling.pathlet_learning import (
    compute_P,
    compute_D0,
    PathletLearning,
)
from modeling.trajectory_encoder import TrajEncoder
from helpers import generate_all_subpaths


def compute_all_edges(trajectories):
    """
    Computes all unique edges from a list of trajectories.
    """
    edges = []
    for p in tqdm(trajectories):
        path_edges = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
        edges += path_edges
    return list(set(edges))


def process_candidate(args):
    """
    Processes a candidate to generate ranked subpaths from their tag trajectories.
    """
    candidate, event, users_tag_ranking, n_traj_per_candidate, lengths, traj_dict = args
    all_trajectories = traj_dict[event]
    trajectories = all_trajectories[candidate]
    user_ranking = users_tag_ranking[candidate[0]]
    adapted_ranking = user_ranking.copy()
    adapted_ranking.insert(0, adapted_ranking.pop(user_ranking.index(candidate[1])))

    sub_paths = []
    index_map = {tag: f"rank_{adapted_ranking.index(tag)}" for tag in adapted_ranking}

    for traj in trajectories[:n_traj_per_candidate]:
        ranked_trajectory = [index_map[tag] for tag in traj]
        sub_paths.extend(generate_all_subpaths(ranked_trajectory, lengths))

    return Counter(sub_paths)


def pool_process_candidates(
    traj_dict, users_tag_ranking, n_traj_per_candidate, lengths, candidate_list, event
):
    """
    Processes multiple candidates in parallel to generate subpath counts.
    """
    args = list(
        zip(
            candidate_list,
            [event] * len(candidate_list),
            [users_tag_ranking] * len(candidate_list),
            [n_traj_per_candidate] * len(candidate_list),
            [lengths] * len(candidate_list),
            [traj_dict] * len(candidate_list),
        )
    )

    with mp.Pool() as pool:
        counters = list(
            tqdm(
                pool.imap(process_candidate, args),
                total=len(candidate_list),
            ),
        )

    grl_counter = Counter()

    for counter in tqdm(counters, desc="Merging Counters"):
        grl_counter.update(counter)

    counters = None

    return grl_counter


def pick_event_trajectories(
    n_traj_for_pathlet_learning, candidate_trajectories, candidates, users_tag_ranking
):
    """
    Randomly selects ranked trajectories from candidate trajectories for pathlet learning.
    """
    trajectories = []
    for _ in tqdm(range(n_traj_for_pathlet_learning)):
        picked_candidate = random.choice(candidates)
        trajectories_from_picked_candidates = candidate_trajectories[picked_candidate]

        picked_index = random.choice(
            list(range(len(trajectories_from_picked_candidates)))
        )
        picked_trajectory = trajectories_from_picked_candidates[picked_index]
        user_ranking = users_tag_ranking[picked_candidate[0]]
        adapted_ranking = user_ranking.copy()
        adapted_ranking.insert(
            0, adapted_ranking.pop(user_ranking.index(picked_candidate[1]))
        )

        ranked_trajectory = [
            f"rank_{adapted_ranking.index(tag)}" for tag in picked_trajectory
        ]

        trajectories.append(tuple(ranked_trajectory))
    return trajectories


def compute_dictionary(
    trajectories,
    sub_paths_counter,
    n_traj_for_pathlet_learning,
    n_candidates,
    learning_parameters,
    users_tag_ranking,
):
    """
    Computes a dictionary for pathlet learning based on selected trajectories and subpaths.
    """

    candidates = list(trajectories.keys())
    picked_trajectories = pick_event_trajectories(
        n_traj_for_pathlet_learning, trajectories, candidates, users_tag_ranking
    )
    candidates_pathlets = [i[0] for i in sub_paths_counter[:n_candidates]]
    all_edges_dict = dict(
        [
            (edge, index)
            for index, edge in enumerate(compute_all_edges(picked_trajectories))
        ]
    )
    P = compute_P(picked_trajectories, all_edges_dict)
    D0 = compute_D0(all_edges_dict, candidates_pathlets)

    print(f"Number of trajectories {len(picked_trajectories)}")
    print(f"Number of candidates pathlets {len(candidates_pathlets)}")
    print(f"Number of edges  {len(all_edges_dict)}")

    dict_computer = PathletLearning(
        P=P,
        D0=D0,
        candidates_pathlets=candidates_pathlets,
        kwargs=learning_parameters,
    )
    dict_computer.compute_dictionary()
    D = dict_computer.D

    return D


def process_candidate_batch(batch, D, users_tag_ranking, trajectories):
    """
    Processes a batch of candidates to compute their embeddings based on ranked trajectories.
    """

    results = []
    for candidate in tqdm(batch):
        user_ranking = users_tag_ranking[candidate[0]]
        adapted_ranking = [candidate[1]] + [
            tag for tag in user_ranking if tag != candidate[1]
        ]
        ranked_trajectories = [
            tuple(f"rank_{adapted_ranking.index(tag)}" for tag in traj)
            for traj in trajectories[candidate]
        ]
        encoder = TrajEncoder(D, ranked_trajectories)
        df = encoder.compute_embeddings()
        candidate_emb = np.mean(df.to_numpy(), axis=0)
        results.append((candidate, candidate_emb))
    return results


def process_batch(args):
    """
    Processes a batch of candidates by calling the candidate embedding function.
    """
    batch, global_D, global_users_tag_ranking, global_trajectories = args
    return process_candidate_batch(
        batch, global_D, global_users_tag_ranking, global_trajectories
    )


def compute_embeddings(D, users_tag_ranking, trajectories, file_path):
    """
    Computes embeddings for candidates in parallel and saves them to a CSV file.
    """

    all_candidates = list(trajectories.keys())
    num_workers = mp.cpu_count()
    batch_size = len(all_candidates) // num_workers

    batches = [
        all_candidates[i : i + batch_size]
        for i in range(0, len(all_candidates), batch_size)
    ]

    args = list(
        zip(
            batches,
            [D] * len(batches),
            [users_tag_ranking] * len(batches),
            [trajectories] * len(batches),
        )
    )

    print(f"num_workers : {num_workers}")
    with mp.Pool(num_workers) as pool:
        results = []
        with tqdm(total=len(all_candidates)) as pbar:
            for batch_results in pool.imap_unordered(process_batch, args):
                results.extend(batch_results)
                pbar.update(len(batch_results))

    df = pd.DataFrame(
        [emb[1] for emb in results],
        index=[emb[0] for emb in results],
        columns=D,
    )
    df.to_csv(file_path)
