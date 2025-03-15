from stcklearn.LinearModel import LinearRegression
from stcklearn.Metrices import mean_squared_error_cus
import numpy as np;

X = np.array([[1], [2], [3], [4], [5]]);
y = np.array([2, 4, 6, 8, 10]);
model = LinearRegression();
model.fit(X,y);


print(model.coef_);
print(model.intercept_)

predict = np.array([1])
print(model.predict(predict))

accuracy = mean_squared_error_cus(y, predict);
print(accuracy);