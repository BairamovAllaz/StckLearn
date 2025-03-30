from sklearn.datasets import load_digits;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support;
from sklearn.metrics import accuracy_score;
from sklearn.preprocessing import StandardScaler;
from matplotlib import pyplot as plt;
import cv2
import numpy as np;

scaler = StandardScaler(); 
mnist = load_digits();

X = mnist.data;
y = mnist.target;


X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=123);

scaler.fit(X_train);
def best_fit_K(): 
    kvals = np.arange(3,100,2);
    accuracys = [];
    for val in kvals:
        model = KNeighborsClassifier(n_neighbors=val)
        model.fit(X_train,y_train);
        pred = model.predict(X_test);
        acc = accuracy_score(y_test, pred);
        accuracys.append(acc);

    accuracy_max = accuracys.index(max(accuracys))
    return accuracy_max

#after exectution of this code we can see model most accurate at K=3;
def pre_procces_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    img_resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    img_normalized = (img_resized * (16 / 255)).astype(np.uint8)
    img_flattened = img_normalized.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flattened)
    print("Processed image array:")
    print(img_normalized)
    print("-" * 40)
    return img_scaled


#print first 5 image 

for i in range(5): 
    print(f"Label : {y[i]}");
    print(X[i].reshape(8,8));
    print("-" * 40)
    
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(mnist.images[i], cmap='gray')
    ax.set_title(f"Label: {mnist.target[i]}")
    ax.axis("off")
plt.show()


model = KNeighborsClassifier(n_neighbors=3);
model.fit(X_train,y_train);

image_path = 'digit.png'

my_image = pre_procces_image(image_path);
prediction = model.predict(X_train);
print(f"Predicted digit: {prediction[0]}");