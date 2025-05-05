"""
Performs model evaluation for next allocation prediction based on 
user behavior trajectories.
"""

import pandas as pd
import numpy as np

from helpers import import_pickle
from evaluating.metrics import make_report
from evaluating.model import build_prediction_for_plug, PreviousPlug
from evaluating.baselines import *

from globals import *

########################################################################

all_users = import_pickle(f"{DEEZER_processed_path}all_users.pkl")
all_tags = import_pickle(f"{DEEZER_processed_path}all_tags.pkl")

X = import_pickle(f"{DEEZER_processed_path}X.pkl")
Y = import_pickle(f"{DEEZER_processed_path}Y.pkl")
X_counts = import_pickle(f"{DEEZER_processed_path}X_counts.pkl")

D_appearance = import_pickle(f"{DEEZER_processed_path}D_appearance.pkl")
D_disappearance = import_pickle(f"{DEEZER_processed_path}D_disappearance.pkl")
appearance_candidates_dict_train = import_pickle(
    f"{DEEZER_processed_path}appearance_candidates_train.pkl"
)
appearance_candidates_dict_test = import_pickle(
    f"{DEEZER_processed_path}appearance_candidates_test.pkl"
)
disappearance_candidates_dict_train = import_pickle(
    f"{DEEZER_processed_path}disappearance_candidates_train.pkl"
)
disappearance_candidates_dict_test = import_pickle(
    f"{DEEZER_processed_path}disappearance_candidates_test.pkl"
)

df_appearance_embeddings_train = pd.read_csv(
    f"{DEEZER_processed_path}appearance_embeddings_train.csv", index_col=0
)
df_appearance_embeddings_test = pd.read_csv(
    f"{DEEZER_processed_path}appearance_embeddings_test.csv", index_col=0
)
df_disappearance_embeddings_train = pd.read_csv(
    f"{DEEZER_processed_path}disappearance_embeddings_train.csv", index_col=0
)
df_disappearance_embeddings_test = pd.read_csv(
    f"{DEEZER_processed_path}disappearance_embeddings_test.csv", index_col=0
)


########################################################################

df_appearance_embeddings_train.columns = D_disappearance
df_disappearance_embeddings_train.columns = D_appearance

Y_train = np.array(
    [[X[(user_id, DEEZER_K - 1, tag)] for tag in all_tags] for user_id in all_users]
)
Y_test = np.array([[Y[(user_id, tag)] for tag in all_tags] for user_id in all_users])

X_prev = np.array(
    [[X[(user_id, DEEZER_K - 2, tag)] for tag in all_tags] for user_id in all_users]
)

########################################################################

baseline = Popularity(all_tags)
baseline.fit(X)
Y_pred = baseline.predict(all_users)
make_report(Y_pred, Y_test, Y_train).to_csv(
    f"{DEEZER_results_path}Popularity.csv", index=False
)

baseline = NMFBaseline(all_tags, X_counts, DEEZER_K)
Y_pred = baseline.predict(all_users)
make_report(Y_pred, Y_test, Y_train).to_csv(
    f"{DEEZER_results_path}NMFBaseline.csv", index=False
)

baseline = Previous(all_tags)
baseline.fit(X)
Y_pred = baseline.predict(all_users)
make_report(Y_pred, Y_test, X_prev).to_csv(
    f"{DEEZER_results_path}Previous.csv", index=False
)

appearance_pred_dict = build_prediction_for_plug(
    df_appearance_embeddings_train,
    df_appearance_embeddings_test,
    appearance_candidates_dict_train,
)
disappearance_pred_dict = build_prediction_for_plug(
    df_disappearance_embeddings_train,
    df_disappearance_embeddings_test,
    disappearance_candidates_dict_train,
)

baseline = PreviousPlug(all_tags, appearance_pred_dict, disappearance_pred_dict)
baseline.fit(X)
Y_pred = baseline.predict(all_users)
make_report(Y_pred, Y_test, Y_train).to_csv(
    f"{DEEZER_results_path}PreviousPlug.csv", index=False
)
