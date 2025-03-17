import numpy as np
class RidgeCus:
    def __init__(self,alpha=1.0) -> None:
        self.coef_ = None;
        self.intercept_ = None;
        self.alpha = alpha;


    def fit(self,X,y): 
        X = np.array(X);
        y = np.array(y);

        X = np.c_[np.ones(X.shape[0]), X];
        I = np.identity(X.shape[1]);
        X_transformX = np.dot(X.T,X);
        alpha_added = X_transformX + self.alpha * I;
        inverse_res = np.linalg.inv(alpha_added);
        X_TransformY = np.dot(X.T,y);
        Beta = np.dot(inverse_res,X_TransformY);
        self.coef_ = round(Beta[0],8);
        self.intercept_ = Beta[1:];

    def predict(self,X):
        X = np.array(X);
        bias_X = np.c_[np.ones(X.shape[0]),X];
        y_predicted = np.dot(bias_X,np.append(self.coef_,self.intercept_));
        return y_predicted;
        

