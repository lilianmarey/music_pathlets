"""
Implements pathlet learning algorithm for trajectory data.
"""

import torch
import numpy as np
from tqdm import tqdm


def compute_P(trajectories, all_edges_dict):
    P = np.zeros((len(all_edges_dict), len(trajectories)))
    for j, p in tqdm(list(enumerate(trajectories))):
        taken_edges = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
        for edge in taken_edges:
            P[all_edges_dict[edge], j] = 1

    return P


def compute_D0(all_edges_dict, candidates_pathlets):
    D0 = np.zeros((len(all_edges_dict), len(candidates_pathlets)))
    for i, p in tqdm(list(enumerate(candidates_pathlets))):
        taken_edges = [(p[j], p[j + 1]) for j in range(len(p) - 1)]
        for edge in taken_edges:
            try:
                D0[all_edges_dict[edge], i] = 1
            except KeyError:
                pass
    return D0


class PathletLearning:
    def __init__(self, P, D0, candidates_pathlets, kwargs):

        self.P_ = torch.from_numpy(P)
        self.D0_ = torch.from_numpy(D0)
        self.alpha_ = torch.from_numpy(np.zeros((D0.shape[1], P.shape[1])))
        self.alpha_.requires_grad = True

        self.candidates_pathlets = candidates_pathlets

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.loss_values_term1 = []
        self.loss_values_term2 = []
        self.loss_values_term3 = []

    def cost_term1(self, alpha):

        value = torch.norm(self.P_ - self.D0_ @ alpha, 2) ** 2 / 2
        return value

    def cost_term2(self, alpha):
        value = torch.norm(alpha, 1)
        return value

    def cost_function(self, alpha):

        v1 = self.cost_term1(alpha)
        v1_value = v1.item()
        self.loss_values_term1.append(v1_value)

        v2 = self.lambda_ * self.cost_term2(alpha)
        v2_value = v2.item()
        self.loss_values_term2.append(v2_value)

        loss_sum = v1 + v2
        self.loss_values.append(v1_value + v2_value)

        return loss_sum

    def build_D(self):

        alpha = self.alpha_.detach().cpu().numpy()
        mean_values = np.mean(alpha, axis=1)

        kept_indices = np.argsort(mean_values)[::-1][: self.dictionary_size]
        D = [self.candidates_pathlets[i] for i in kept_indices]
        self.D = D

    def compute_dictionary(self):

        optimizer = torch.optim.Adam([self.alpha_], lr=0.1)

        self.loss_values = []
        for i in tqdm(range(self.n_steps)):
            loss = self.cost_function(self.alpha_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.alpha_.data.clamp_(0, 1)
            if i > 10 and np.mean(self.loss_values[-10:-5]) <= np.mean(
                self.loss_values[-5:]
            ):
                break

        self.build_D()
