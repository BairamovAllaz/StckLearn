import numpy as np

def mean_squared_error_cus(y_true,y_predicted):
    """
    Compute the Mean Squared Error (MSE) between true values and predicted values.
    
    Parameters:
    y_true (array-like): Actual target values
    y_pred (array-like): Predicted values from the model
    
    Returns:
    float: The mean squared error
    """
    y_true = np.array(y_true);
    y_predicted = np.array(y_predicted);
    
    #calculation
    residual = y_predicted - y_true;
    squared_residual = residual**2;
    mse = np.mean(squared_residual);
    return mse


