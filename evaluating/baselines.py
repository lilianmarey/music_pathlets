import numpy as np
from tqdm import tqdm
from sklearn.decomposition import NMF
from globals import *


class Popularity:
    """
    A class that implements a popularity-based prediction model.
    """
    def __init__(self, all_tags):
        self.all_tags = all_tags

    def fit(self, X):
        self.X = X
        self.tag_pop_dict = dict([(tag, 0) for tag in self.all_tags])

        for key, val in tqdm(list(X.items())):
            if key[1] != DEEZER_K - 1:
                self.tag_pop_dict[key[2]] += val

    def predict(self, user):

        pred = np.array(
            [[self.tag_pop_dict[tag] for tag in self.all_tags] for _ in user]
        )

        for user_index in range(len(user)):
            pred[user_index, :] /= np.sum(pred[user_index, :])

        return pred


class Previous:
    """
    A class that implements a previous interaction-based prediction model.
    """
    def __init__(self, all_tags):
        self.all_tags = all_tags

    def fit(self, X):
        self.X = X
        X_keys = list(X.keys())
        self.max_period = np.max([key[1] for key in X_keys])

    def predict(self, test_users):

        pred = np.array(
            [
                [self.X[(user_id, self.max_period, tag)] for tag in self.all_tags]
                for user_id in test_users
            ]
        )
        return pred


class NMFBaseline:
    """
    A class that implements a Non-negative Matrix Factorization (NMF) baseline model 
    for prediction based on user interactions.
    """
    def __init__(self, all_tags, X_counts, max_period):
        self.all_tags = all_tags
        self.X_counts = X_counts
        self.max_period = max_period

    def build_interaction_matrix(self, users):
        interaction_matrix = np.zeros((len(users), len(self.all_tags)))

        for user_id in tqdm(users):
            for tag in self.all_tags:
                v = np.mean(
                    [self.X_counts[(user_id, t, tag)] for t in range(self.max_period)]
                )
                interaction_matrix[users.index(user_id), self.all_tags.index(tag)] += v
        self.interaction_matrix = interaction_matrix

    def predict(self, test_users):
        self.build_interaction_matrix(test_users)
        model = NMF(n_components=50, init="random", random_state=13, max_iter=10000)
        W = model.fit_transform(self.interaction_matrix)
        H = model.components_
        Y_pred = W @ H
        for user_index in range(len(test_users)):
            Y_pred[user_index, :] /= np.sum(Y_pred[user_index, :])

        return Y_pred
