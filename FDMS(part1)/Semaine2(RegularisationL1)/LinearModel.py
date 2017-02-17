import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

def gradient_stochastique(X,Y,lamb,epsilone,I):
	"""
	gradient stochastique with gradient clipping

	X : Data for prediction
	Y : variable to predict
	lamb : coefficient of regularisation
	epsilone : pas du gradient
	I : nombre d'itération
	""" 

	# l is number of samples
    l=X.shape[0]

    # n is number of features
    n=X.shape[1]

    #initialisation du gradient
    grad1 = np.random.randn(n,1)
    grad2 = np.random.randn(n,1)

    #Demarrer Iteration
    for it in range(I):
        for i in range(n):

        	#get one sample for iteration
            idx = np.random.randint(l)
            error = Y[idx]-X[idx,:].dot(grad1)
            A = np.array([2*epsilone*(X[idx,:])*error]).T
            grad2=grad1+A-lamb*np.sign(grad1)

            #gradient clipping
            for j in range(n):
                if grad2[j,0]*grad1[j,0]<0:
                    grad1[j,0]=0
                else:
                    grad1[j,0]=grad2[j,0]
    return grad1


class RegularisationL1(BaseEstimator,ClassifierMixin):
    """Régularisation L1"""

    def __init__(self,lamb,eps,I):
        self.lamb=lamb
        self.eps=eps
        self.I=I

    def fit(self,X,Y):
        #l correspond aux nombres de données dans notre ensemble d'apprentissage
        #n correspond aux nombres de variables de chaques vecteurs
        l,n=X.shape
        self.data = np.random.randn(self.I,2)
        
        #On initialise les poids
        grad1 = np.random.randn(n,1)
        grad2 = np.random.randn(n,1)
        
        #Calcul du gradient
        for it in range(self.I):
            for i in range(n):
                idx = np.random.randint(l)
                error = Y[idx]-X[idx,:].dot(grad1)
                A = np.array([2*self.eps*(X[idx,:])*error]).T
                grad2=grad1+A-self.lamb*np.sign(grad1)
                for j in range(n):
                    if grad2[j,0]*grad1[j,0]<0:
                        grad1[j,0]=0
                    else:
                        grad1[j,0]=grad2[j,0]
                        
            Ltheta = (1/float(l))*((Y-X.dot(grad1))**2).sum()+self.lamb*(abs(grad1)).sum()    
            self.coef=grad1
            self.data[it,0]=Ltheta
            self.data[it,1]= accuracy_score(self.predict(X),Y)
        return self

    def predict(self,X):
        l=X.shape[0]
        result = X.dot(self.coef)
        result = np.where(result>0.5,1,0)
        return result

class RegularisationL2(BaseEstimator,ClassifierMixin):
    """Régularisation L2"""

    def __init__(self,lamb,eps,I):
        self.lamb=lamb
        self.eps=eps
        self.I=I

    def fit(self,X,Y):
        #l correspond aux nombres de données dans notre ensemble d'apprentissage
        #n correspond aux nombres de variables de chaques vecteurs
        l,n=X.shape
        self.data = np.random.randn(self.I,2)
        
        #On initialise les poids
        grad1 = np.random.randn(n,1)
        grad2 = np.random.randn(n,1)
        
        #Calcul du gradient
        for it in range(self.I):
            for i in range(n):
                idx = np.random.randint(l)
                error = Y[idx]-X[idx,:].dot(grad1)
                A = np.array([2*self.eps*(X[idx,:])*error]).T
                grad2=grad1+A-2*self.lamb*np.sign(grad1)*grad1
                for j in range(n):
                    if grad2[j,0]*grad1[j,0]<0:
                        grad1[j,0]=0
                    else:
                        grad1[j,0]=grad2[j,0]
            Ltheta = (1/float(l))*((Y-X.dot(grad1))**2).sum()+self.lamb*(abs(grad1)).sum()    
            self.coef=grad1
            self.data[it,0]=Ltheta
            self.data[it,1]= accuracy_score(self.predict(X),Y)
        return self

    def predict(self,X):
        l=X.shape[0]
        result = X.dot(self.coef)
        result = np.where(result>0.5,1,0)
        return result

class RegularisationL1_L2(BaseEstimator,ClassifierMixin):
    """Régularisation combinaison L1 et L2"""

    def __init__(self,lamb1,lamb2,eps,I):
        self.lamb1=lamb1
        self.lamb2=lamb2
        self.eps=eps
        self.I=I

    def fit(self,X,Y):
        #l correspond aux nombres de données dans notre ensemble d'apprentissage
        #n correspond aux nombres de variables de chaques vecteurs
        l,n=X.shape
        self.data = np.random.randn(self.I,2)
        
        #On initialise les poids
        grad1 = np.random.randn(n,1)
        grad2 = np.random.randn(n,1)
        
        #Calcul du gradient
        for it in range(self.I):
            for i in range(n):
                idx = np.random.randint(l)
                error = Y[idx]-X[idx,:].dot(grad1)
                A = np.array([2*self.eps*(X[idx,:])*error]).T
                grad2=grad1+A-self.lamb1*np.sign(grad1)-2*self.lamb2*np.sign(grad1)*grad1
                for j in range(n):
                    if grad2[j,0]*grad1[j,0]<0:
                        grad1[j,0]=0
                    else:
                        grad1[j,0]=grad2[j,0]
            Ltheta = (1/float(l))*((Y-X.dot(grad1))**2).sum()+self.lamb1*(abs(grad1)).sum()+self.lamb2*(grad1*grad1).sum()    
            self.coef=grad1
            self.data[it,0]=Ltheta
            self.data[it,1]= accuracy_score(self.predict(X),Y)
        return self

    def predict(self,X):
        l=X.shape[0]
        result = X.dot(self.coef)
        result = np.where(result>0.5,1,0)
        return result