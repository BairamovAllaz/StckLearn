import numpy as np
from stcklearn.LinearModel import RidgeCus;
from stcklearn.SVM import SvcCus;

X = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])  # Feature matrix
y = np.array([1, 1, -1, -1])  # Labels
model = SvcCus();
model.fit(X,y);

new_data = np.array([[2, 2], [1, 1]])
prediction = model.predict(new_data);
print("Predicted: " , prediction);
