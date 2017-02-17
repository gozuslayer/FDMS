import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.svm import SVC
from sklearn import cross_validation



class RandomOuSvmClassifier(BaseEstimator,ClassifierMixin):
	def __init__(self,nature="random"):
		self.nature=nature
		self.svm=SVC()
	def fit(self, X, y):
		if(self.nature=="svm"):
			self.svm.fit(X, y)
		return self
	def predict(self, X):
		if(self.nature=="random"):
			return np.random.randint(0,2,len(X))
		else:
			return self.svm.predict(X)




###################Main####

from sklearn import datasets
from sklearn import metrics


iris = datasets.load_iris()
X=iris.data
y=iris.target
classifieurRandom= RandomOuSvmClassifier(nature="random")
classifieurSvm= RandomOuSvmClassifier(nature="svm")
scoresRandom = cross_validation.cross_val_score(classifieurRandom, X, y, cv=15
,scoring="accuracy")
scoresSvm = cross_validation.cross_val_score(classifieurSvm, X, y, cv=15
,scoring="accuracy")

print scoresRandom,scoresSvm
