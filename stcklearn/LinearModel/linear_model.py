import numpy as np;

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    # The goal of the fit() function in linear regression is to learn the
    # relationship between the features X and the target variable y.
    # This is done by finding the coefficients (weights) 𝑤 w and the intercept 𝑏
    # that minimize the error between the predicted and actual values
    # The model assumes a linear relationship between 𝑋 and y
    # such that: y=Xw+b
    def fit(self,X,y):
        X = np.array(X);
        y = np.array(y);
        #adding bias term to the X
        X = np.c_[np.ones(X.shape[0]),X];
        # Calculate (X^T * X)
        product_T_X = np.dot(np.transpose(X),X);
        # Calculate (X^T * y)
        product_TX_y = np.dot(np.transpose(X),y);
        # Inverse of (X^T * X)
        inversed_product_T_X = np.linalg.inv(product_T_X);
        # Calculate the coefficient vector (including intercept and coefficients)
        CoefficientW = np.dot(inversed_product_T_X,product_TX_y);
        self.coef_ = round(CoefficientW[0],8)  # w Coefficient and rounding to decimal point 8
        self.intercept_ = CoefficientW[1:]; #b intercept
        # so the model correctly learns  at y=2x from y = wX+b
    
    def predict(self,X):
        X = np.array(X);
        bias_X = np.c_[np.ones(X.shape[0]),X];
        y_predicted = np.dot(bias_X,np.append(self.coef_,self.intercept_));
        return y_predicted;



        