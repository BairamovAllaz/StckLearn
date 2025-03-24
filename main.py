import numpy as np
import pandas as pd;
from stcklearn.Tree import DecisionTreeClassifier
from IPython.display import display
from sklearn.model_selection import train_test_split


col_names = ['sepal_length','sepal_width','petal_lenght','petal_width','type'];

data = pd.read_csv('Iris.csv',skiprows=1,header=None,names=col_names);
display(data.head(10));


model = DecisionTreeClassifier();
model.fit()
