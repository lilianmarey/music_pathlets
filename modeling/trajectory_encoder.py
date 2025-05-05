"""
Defines the TrajEncoder class, which encodes trajectories 
using a dictionary of pathlets.
"""

import pandas as pd
import numpy as np
from helpers import generate_all_subpaths


class TrajEncoder:
    def __init__(self, D, trajectories):
        self.D = D
        self.trajectories = trajectories

    def cut_path(self, traj, pathlet):
        indices = [index for index, node in enumerate(traj) if node == pathlet[0]]
        for index in indices:
            if traj[index : index + len(pathlet)] == pathlet:
                break
        assert traj[index : index + len(pathlet)] == pathlet
        prefix = traj[: index + 1]
        suffix_start = index + len(pathlet)
        suffix = traj[suffix_start - 1 :]
        if len(prefix) == 1:
            prefix = ()
        if len(suffix) == 1:
            suffix = ()
        return tuple(prefix), tuple(suffix)

    def recursive_encode_path(self, traj, dictionary):
        subpath_lengths = list(set([len(pathlet) for pathlet in dictionary]))
        if len(traj) == 0:
            return []
        else:
            sub_paths = set(generate_all_subpaths(traj, subpath_lengths))
            candidates_in_dict = set(dictionary).intersection(sub_paths)
            if len(candidates_in_dict) == 0:
                return []
            longest_pathlet = sorted(
                list(candidates_in_dict), key=lambda pathlet: -len(pathlet)
            )[0]
            p1, p2 = self.cut_path(traj, longest_pathlet)
            return (
                self.recursive_encode_path(p1, candidates_in_dict)
                + [longest_pathlet]
                + self.recursive_encode_path(p2, candidates_in_dict)
            )

    def compute_embeddings(self):
        all_pathlets_dict = dict([(i[1], i[0]) for i in enumerate(self.D)])
        embeddings = []
        for p in self.trajectories:
            reconstruction = self.recursive_encode_path(p, self.D)
            activated_pathlet_indices = [
                all_pathlets_dict[pathlet] for pathlet in reconstruction
            ]

            p_vector = [0 for _ in range(len(self.D))]
            for index in activated_pathlet_indices:
                p_vector[index] += 1

            embeddings.append(p_vector)
        embeddings_df = pd.DataFrame(
            np.array(embeddings), columns=self.D, index=self.trajectories
        )
        return embeddings_df
