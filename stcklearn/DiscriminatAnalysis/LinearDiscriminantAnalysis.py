import numpy as np

"Steps to take mathematicaly"
# 1  Compute Class Means
# 2 Compute Overall Mean
# 3 Compute Between-Class Scatter Matrix SB
# 4  Compute Within-Class Scatter Matrix
# 5 Compute SW^-1*SB
# 6 Find Eigenvectors and Eigenvalues
# predict using Euclidean distance algorithm


class LinearDiscriminantAnalysisCustom:
    def __init__(self) -> None:
        self.eigenvector = None;
        self.class_means = None;

    def fit(self,X,y):
        X = np.array(X);
        y = np.array(y);
        
        number_of_classes = len(np.unique(y));  
        self.class_means = {};
        for n in range(number_of_classes): 
            self.class_means[n] = np.mean(X[y==n],axis=0);

        overall_mean = np.mean(X, axis=0);

        SB = np.zeros((X.shape[1],X.shape[1]));
        for c in np.unique(y): 
            N_c = np.sum(y==c);
            diff = (self.class_means[c] - overall_mean).reshape(-1,1);
            SB += N_c * np.dot(diff,diff.T);

        SW = np.zeros((X.shape[1],X.shape[1]));
        for c in range(number_of_classes): 
            X_c = X[y==c];
            X_c_centerd = X_c - self.class_means[c];
            SW += np.dot(X_c_centerd.T, X_c_centerd);
        
        Sw_reverse = np.linalg.inv(SW);
        SwSB = np.dot(Sw_reverse,SB);

        eigenvalue, eigenvector = np.linalg.eig(SwSB)
        idx = eigenvalue.argsort()[::-1]
        self.eigenvector = eigenvector[:, idx]

    def predict(self,X):
        X = np.array(X);
        if self.eigenvector is None: 
            raise ValueError("Model not fitted. Call `fit` first.")
        projected = np.dot(X,self.eigenvector)
        predictions = []
        for sample in projected:
            distances = {}
            for c, mean in self.class_means.items():
                projected_mean = np.dot(mean, self.eigenvector)
                distances[c] = np.linalg.norm(sample - projected_mean)
            predicted_class = min(distances, key=distances.get)
            predictions.append(predicted_class)
        return np.array(predictions);
