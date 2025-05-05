import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def TV(y1, y2):
    """
    Compute the total variation distance between two 1-D allocations.
    """

    y1, y2 = np.array(y1), np.array(y2)
    return np.sum(np.abs(y1 - y2)) / 2


def ATV(Y1, Y2):
    """
    Calculate the average total variation distance between two sets of 1-D allocations.
    """
    return np.mean([TV(y1, y2) for y1, y2 in list(zip(Y1, Y2))])


def plus_minus_sub(y_pred, y_test, x_last):
    """
    This function evaluates whether each element in the predicted and actual test
    values indicates an increase or decrease relative to the last known values.
    """

    observed_change = []
    for i in range(len(y_test)):
        if y_test[i] >= x_last[i]:
            observed_change.append(1)
        elif y_test[i] < x_last[i]:
            observed_change.append(0)

    predicted_change = []
    for i in range(len(y_pred)):
        if y_pred[i] >= x_last[i]:
            predicted_change.append(1)
        elif y_pred[i] < x_last[i]:
            predicted_change.append(0)

    return np.array(observed_change), np.array(predicted_change)


def plus_minus(Y_pred, Y_test, X_last):
    """
    This function aggregates the observed and predicted changes for each user and computes
    the ROC AUC score based on the combined results.
    """

    all_observed = []
    all_predicted = []

    for user_index in range(len(Y_pred)):
        y_pred, y_test, x_last = (
            Y_pred[user_index],
            Y_test[user_index],
            X_last[user_index],
        )
        observed_change, predicted_change = plus_minus_sub(y_pred, y_test, x_last)
        all_observed.extend(list(observed_change))
        all_predicted.extend(list(predicted_change))

    try:
        return roc_auc_score(all_predicted, all_observed)
    except:
        return "Error"


def new_class_t_sub(y_pred, y_test, x_last):
    """
    This function evaluates whether each element in the predicted and actual test values
    indicates a change (greater than zero) only when the corresponding last known value is zero.
    """

    observed_change = []
    predicted_change = []
    for i in range(len(y_test)):
        if x_last[i] == 0:
            observed_change.append(int(y_test[i] > 0))
            predicted_change.append(int(y_pred[i] > 0))

    return np.array(observed_change), np.array(predicted_change)


def new_class_t(Y_pred, Y_test, X_last, metric=roc_auc_score):
    """
    This function aggregates the observed and predicted changes for each user
    and computes the specified performance metric based on the combined results.
    """

    all_observed = []
    all_predicted = []

    for user_index in range(len(Y_pred)):
        y_pred, y_test, x_last = (
            Y_pred[user_index],
            Y_test[user_index],
            X_last[user_index],
        )
        observed_change, predicted_change = new_class_t_sub(y_pred, y_test, x_last)
        all_observed.extend(list(observed_change))
        all_predicted.extend(list(predicted_change))

    try:
        return metric(all_predicted, all_observed)
    except:
        return "Error"


def make_report(Y_pred, Y_test, X_last):
    """
    Generate a report summarizing various performance metrics based on predicted and observed values.
    """
    report_df = []
    report_df.append(["ATV", ATV(Y_pred, Y_test)])
    report_df.append(["plus_minus", plus_minus(Y_pred, Y_test, X_last)])
    report_df.append(
        ["new_class_t_roc", new_class_t(Y_pred, Y_test, X_last, metric=roc_auc_score)]
    )
    report_df = pd.DataFrame(report_df, columns=["metric", "value"])
    return report_df
