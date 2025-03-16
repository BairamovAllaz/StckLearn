from stcklearn.DiscriminatAnalysis import LinearDiscriminantAnalysisCustom
from stcklearn.Metrices import accuracy_score_cus;
import numpy as np;


X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 8],
    [7, 8],
    [8, 9]
])
y = np.array([0, 0, 0, 1, 1, 1])


model = LinearDiscriminantAnalysisCustom()
model.fit(X,y);

predictions = model.predict(X);
print(predictions);

score = accuracy_score_cus(y,predictions);
print(score);