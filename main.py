from stcklearn.LinearModel import RidgeCus;

X = [[1], [2], [3]]
y = [2, 4, 6]

model = RidgeCus(alpha=1.0);
model.fit(X,y);
print(model.coef_);
print(model.intercept_);


X_test = [[1], [5]]
predicted = model.predict(X_test);
print("predicted: ",predicted);
