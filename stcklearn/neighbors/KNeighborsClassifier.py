import numpy as np;
class KNeighborsClassifier: 
    def __init__(self, n_neighbors = 3) -> None:
        self.n_neighbors = n_neighbors;
        self.X_train = None;
        self.Y_train = None;

    def fit(self, X ,y): 
        X = np.array(X);
        y = np.array(y);    
        self.X_train = X;
        self.Y_train = y;

    def Euclidean_distance(self,A,B):
        return np.sqrt(np.sum(A-B)**2);

    def majority_voting(self,distances): 
        votes = [label for _, label in distances[:self.n_neighbors]];
        votes_dic = {} #dictionary 

        for vote in votes: 
            if vote in votes_dic:
                votes_dic[vote] += 1;
            else:
                votes_dic[vote] = 1
        return max(votes_dic,key=votes_dic.get);


    def predict(self, pred):
        pred = np.array(pred);  
        if self.X_train is None  or self.Y_train is None:
            print("Please fit the model~!!!!");
            return;
        distances = [];
        for i in range(len(self.X_train)): 
            distances.append((self.Euclidean_distance(pred,self.X_train[i]),self.Y_train[i]));
        distances.sort();
        predicted = self.majority_voting(distances);
        return predicted;