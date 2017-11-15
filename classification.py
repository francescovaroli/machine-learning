import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""
    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))

class LabelPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""
    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        label = [0 for n in range(len(y))]
        for brain in range(len(y)):
            if y[brain, 0] >= 0.7473:
                label[brain] = 0
            if y[brain, 1] >= 0.2689:
                label[brain] = 1
            if y[brain, 2] >= 0.2558:
                label[brain] = 2
            if y[brain, 3] >= 0.3659:
                label[brain] = 3


        self.treeClassifier = GaussianNB()
        self.treeClassifier.fit(X,label)
        return self

    def predict_proba(self, X):
        proba = self.treeClassifier.predict_proba(X)
        for i in range(1, len(proba)):
            proba[i][0] = (0.7218 + proba[i][0])/2
            proba[i][1] = (0.1761 + proba[i][1])/2
            proba[i][2] = (0.0732 + proba[i][2])/2
            proba[i][3] = (0.0289 + proba[i][3])/2
        return proba







class MultiDTC():
    def fit(self, X, y):
        self.treeClassifier = [0,0,0,0]
        for index in range(4):
            self.treeClassifier[index] = Ridge()
            label = y[:,index]
            self.treeClassifier[index].fit(self, X, label)
        return self

    def predict_proba(self, X):
        l = []
        for i in range(4):
            l[i]=self.treeClassifier[i].predict(X)
        for brain in range(len(X)):
            for index in range(4):
                l[index][brain] = l[index]/(l[0][brain] + l[1][brain] + l[2][brain]+ l[3][brain])
        return l

