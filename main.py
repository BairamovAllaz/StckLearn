import numpy as np
import pandas as pd;
from stcklearn.neighbors import KNeighborsClassifier;
import matplotlib.pyplot as plt

model = KNeighborsClassifier();

X_train = np.array([[150, 1], [170, 1], [130, 0], [120, 0], [140, 0]])
y_train = np.array(['A', 'A', 'O', 'O', 'O'])

# Test point
X_test = np.array([145, 1])


model.fit(X_train, y_train);
print("Predicted: " , model.predict(X_test));


plt.scatter(X_test[0], X_test[1], color='blue',
            marker='x', s=100, label='Test Point')
plt.xlabel("Weight (grams)")
plt.ylabel("Smoothness (1 = Smooth, 0 = Rough)")
plt.legend()
plt.title("KNN Classification of Fruit (2D Projection)")
plt.grid(True)
plt.show()
