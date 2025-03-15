import numpy as np


def sigmoid(log_odd):
    return 1 / (1 + np.exp(-log_odd))

class LogisticRegressionCustom:
    def __init__(self):
        self.beta_0 = 0
        self.beta_1 = 0

    def fit(self, X, y, learning_rate=0.001, iterations=5000):
        m = len(y)
        X = np.array(X);
        for i in range(iterations):
            log_odds = self.beta_0 + self.beta_1 * X
            y_pred = sigmoid(log_odds)
            error = y_pred - y
            # d_beta_0 is equal to sum of errors divided by number if samples
            d_beta_0 = np.sum(error) / m
            # d_beta_1 is equal to sum of errors multiplied by X divided by number if samples
            d_beta_1 = np.sum(error * X) / m
            # Gradient Descent 
            self.beta_0 = self.beta_0 - (learning_rate * d_beta_0)
            self.beta_1 = self.beta_1 - (learning_rate * d_beta_1)


    def predict(self, X):
        X = np.array(X);
        liner_combination = self.beta_0 + self.beta_1 * X
        y_pred_prob = sigmoid(liner_combination)
        y_pred = np.round(y_pred_prob)
        return y_pred
