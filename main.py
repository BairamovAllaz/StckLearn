from stcklearn.LinearModel import LogisticRegressionCustom
import numpy as np;

X = np.array([1, 2, 3, 4, 5])  # Feature (input)
y = np.array([0, 0, 0 , 1,1])  # Target (binary outcome)


x_topr = np.array([6])
model = LogisticRegressionCustom();
model.fit(X,y);
predicted = model.predict(x_topr)
print(predicted); 