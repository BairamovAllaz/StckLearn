import numpy as np
from scipy.optimize import minimize


class SvcCus:
    def __init__(self) -> None:
        self.w = None
        self.bias = None
        self.support_vectors = None
        self.support_labels = None
        self.support_alphas = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        def objective(alpha):
            first_term = sum(
                (alpha[i] * alpha[j] * y[i] * y[j]) * np.dot(X[i], X[j])
                for i in range(n_samples)
                for j in range(n_samples)
            )
            return 0.5 * first_term - np.sum(alpha)

        def Constraints(alpha):
            return np.sum(alpha*y)

        bounds = [(0, None) for _ in range(n_samples)]

        constraints = {'type': 'eq', 'fun': Constraints}
        alpha_initial = np.zeros(n_samples)

        result = minimize(objective, alpha_initial,
                          bounds=bounds, constraints=constraints)
        alpha = result.x

        support_vector_indices = alpha > 1e-5
        self.support_vectors = X[support_vector_indices]
        self.support_labels = y[support_vector_indices]
        self.support_alphas = alpha[support_vector_indices]

        self.w = np.sum(
            self.support_alphas[i]*self.support_labels[i] *
            self.support_vectors[i]
            for i in range(len(self.support_vectors))
        )

        self.bias = np.mean([
            self.support_labels[i] - np.dot(self.w, self.support_vectors[i])
            for i in range(len(self.support_vectors))
        ])

    def predict(self, X):
        X = np.array(X)
        if self.w is None or self.bias is None:
            raise ValueError("Model has not been trained. Call `fit` first.")
        predicted = np.sign(np.dot(X, self.w) + self.bias)
        return predicted
