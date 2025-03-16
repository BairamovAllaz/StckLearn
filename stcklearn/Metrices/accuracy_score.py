import numpy as np


def accuracy_score_cus(y_actual, y_predicted):
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    if len(y_actual) != len(y_predicted):
        raise ValueError("Length of y_true and y_pred must be the same.")
    correct = sum(y_actual == y_predicted)
    accuracy = correct / len(y_actual)
    return accuracy
