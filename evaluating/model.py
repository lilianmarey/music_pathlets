import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from ast import literal_eval

from globals import *


def build_prediction_for_plug(
    df_embeddings_train,
    df_embeddings_test,
    candidates_dict_train,
):
    """
    Build a prediction model using Random Forest classifier and return predicted events (appearance or disappearance).
    """


    output_train = [
        candidates_dict_train[literal_eval(i)] for i in list(df_embeddings_train.index)
    ]

    class_model = RandomForestClassifier

    model = class_model()
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, 30],
        "min_samples_split": [
            2,
            5,
            10,
        ],
        "min_samples_leaf": [
            1,
            2,
            4,
        ],
        "bootstrap": [True],
    }

    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="precision", n_jobs=-1, verbose=2
    )

    grid_search.fit(df_embeddings_train.to_numpy(), output_train)
    print(grid_search.best_params_)
    model = class_model(**grid_search.best_params_).fit(
        df_embeddings_train.to_numpy(), output_train
    )

    event_pred = model.predict(df_embeddings_test.to_numpy())

    event_pred_dict = dict(
        list(
            zip(list(map(literal_eval, df_embeddings_test.index.tolist())), event_pred)
        )
    )

    return event_pred_dict


class PreviousPlug:
    """
    This class predicts the next allocation based on last allocation, plugging appearance 
    and disappearance predictions to adjust the final rates.
    """
    def __init__(self, all_tags, appearance_pred_dict, disappearance_pred_dict):
        self.all_tags = all_tags
        self.appearance_pred_dict = appearance_pred_dict
        self.disappearance_pred_dict = disappearance_pred_dict

    def fit(self, X):
        self.X = X
        X_keys = list(X.keys())
        self.max_period = np.max([key[1] for key in X_keys])

    def predict(self, users):

        pred = np.array(
            [
                [self.X[(user_id, self.max_period, tag)] for tag in self.all_tags]
                for user_id in users
            ]
        )
        for user_index in range(len(users)):
            for tag_index in range(len(self.all_tags)):
                if pred[user_index, tag_index] > 0:
                    if (
                        self.disappearance_pred_dict[
                            (users[user_index], self.all_tags[tag_index])
                        ]
                    ) == 1:
                        pred[user_index, tag_index] = 0
                if pred[user_index, tag_index] == 0:
                    try:
                        if (
                            self.appearance_pred_dict[
                                (users[user_index], self.all_tags[tag_index])
                            ]
                        ) == 1:
                            consumptions = [
                                self.X[
                                    (
                                        users[user_index],
                                        t,
                                        self.all_tags[tag_index],
                                    )
                                ]
                                for t in range(self.max_period)
                            ]
                            pred[user_index, tag_index] = np.mean(
                                [v for v in consumptions if v > 0]
                            )
                    except KeyError:
                        pass

        for user_index in range(len(users)):
            if np.sum(pred[user_index, :]) > 0:
                pred[user_index, :] /= np.sum(pred[user_index, :])
            else:
                pred[user_index, :] = np.array(
                    [
                        self.X[(users[user_index], self.max_period, tag)]
                        for tag in self.all_tags
                    ]
                )

        return pred
