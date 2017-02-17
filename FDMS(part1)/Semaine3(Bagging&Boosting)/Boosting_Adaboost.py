%matplotlib inline 
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import tree
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt


class bagging():
    """Bagging de classifieur binaire label 1 : -1 """

    def __init__(self,B=5):
        """Constructeur
        B: Nombre de classifieur a considere"""

        self.B = B   
   
    def fit(self, X,Y):
        """Training"""

        # n  correspond au nombre de sample
        n = X.shape[0] 

        #Bootstrap pour mes classifieurs
        n_class = n/self.B 

        #initialisation des classifieurs
        self.clf = [tree.DecisionTreeClassifier(max_depth=1) for i in range(self.B)] 

        #training de chaque classifieur sur le bootstrap
        for i in range(self.B): 

            #Bootstrap pour mes classifieurs
            ids = np.random.choice(n,n_class) 
            self.clf[i]= self.clf[i].fit(X[ids],Y[ids]) 
        
    def prediction(self, X):
        out = []
        for i in range(len(X)): 
            result = []
            for j in range(self.B): 
                result.append(self.clf[j].predict(X[i].reshape(1, X.shape[1]))) 

            #On choisit celui qui a le plus de vote (en cas d'égalité on met positif)
            out.append(np.sign(np.sum(result))) 
        return np.array(out)

    def accuracy(self, pred , Y):
        acc = pred - Y
        return np.where(acc==0,1,0).sum()/len(pred) 



breast_cancer_scale = datasets.fetch_mldata('breast-cancer_scale')
X=breast_cancer_scale.data
Y=breast_cancer_scale.target
Y[np.where(Y == 4)] = 1
Y[np.where(Y == 2)] = -1
xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

maxc=50
acc=np.zeros(maxc)
training_error=np.zeros(maxc)
for B in xrange(0,maxc):
    bag = bagging(B+1)
    train = bag.fit(xtrain,ytrain)

    pred = bag.prediction(xtrain)                      #Estimation de l'erreur sur la base d'apprentissage
    training_error[B]= 100-bag.accuracy(pred,ytrain)
    
    pred = bag.prediction(xtest)                       #Estimation du taux de reconnaissance sur la base de test
    acc[B]= bag.accuracy(pred,ytest)
    print('%2d classifiers =>    Accuracy:%.2f%%      Training Error Rate:%.2f%%' % (B+1,acc[B],training_error[B]))

    
plt.plot(np.linspace(1,maxc,maxc),training_error)
plt.xlabel('Classifier number')
plt.ylabel('Training Error Rate')
plt.title('Bagging - Decision Tree')
plt.show()
    
plt.plot(np.linspace(1,maxc,maxc),acc)
plt.xlabel('Classifier number')
plt.ylabel('Accuracy')
plt.title('Bagging - Decision Tree')
plt.show()

from __future__ import division
import math
class Boost():
    '''Boosting with classifieur binaire label 1:-1 '''
    
    def __init__(self, B):
        #B : number of classifier
        self.B = B 
        
    def fit(self, x, y):
        
        #Initialisation des poids 
        self.w = np.ones((len(x)))/len(x)
        
        #Initialisation des classifieurs
        self.classifs = [tree.DecisionTreeClassifier(max_depth=1) for i in range(self.b)]

        #Initialisation des alphas
        self.alphas = [0 for i in range(self.b)]

        #
        for i in range(self.b):
            self.classifs[i].fit(x , y , sample_weight = self.w) 
            pred = self.classifs[i].predict(x) 

            #On recupere les samples mal prédit
            idx_diff = np.where(pred != y)[0] 
            
            error = 0
            for idx in idx_diff:
                error += self.w[idx]
            
            #On recupere seulement quand lerreur de prédiction est inférieur à 0.5
            if error < 0.5 :
                self.alphas[i] = float(0.5*np.log((1 - error) / error))

                # Z facteur de normalisation
                z = float(2*np.sqrt(error*(1-error)))
                
                #mettre a jour des poids des samples                    
                for k in range(len(self.w)): 
                    self.w[k] *= (np.exp(-self.alphas[i] * y[k] * pred[k]) / (z*1.0))

            else : 
                self.alpha[i]= 0  
        
        
    def predict(self, x):
        pred = []
        for example in x:
            results = np.array([self.classifs[i].predict(example.reshape(1 , x.shape[1])) for i in range(self.b)])

            final = np.sign(np.dot(np.array(self.alphas).T , results))
            pred.append(final)
        return np.array(pred) 
    
    def accuracy(self, pred , Y):
        acc = pred - Y.reshape(len(Y),1)
        return np.where(acc==0,1,0).sum()/len(pred)


breast_cancer_scale = datasets.fetch_mldata('breast-cancer_scale')
X=breast_cancer_scale.data
Y=breast_cancer_scale.target
Y[np.where(Y == 4)] = 1
Y[np.where(Y == 2)] = -1
xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)    

maxc=50
acc=np.zeros(maxc)
training_error=np.zeros(maxc)
for B in xrange(0,maxc):
    boost = Boost(B+1)
    boost.fit(xtrain , ytrain)
    
    pred = boost.predict(xtrain)                        #Estimation de l'erreur sur la base d'apprentissage  
    training_error[B]=100-boost.accuracy(pred,ytrain)
    
    pred = boost.predict(xtest)                         #Estimation du taux de reconnaissance sur la base de test 
    acc[B]= boost.accuracy(pred,ytest)
    print('%2d classifiers =>    Accuracy:%.2f%%      Training Error Rate:%.2f%%' % (B+1,acc[B],training_error[B]))

plt.plot(np.linspace(1,maxc,maxc),training_error)
plt.xlabel('Classifier number')
plt.ylabel('Training Error Rate')
plt.title('AdaBoost - Decision Tree')
plt.show()

plt.plot(np.linspace(1,maxc,maxc),acc)
plt.xlabel('Classifier number')
plt.ylabel('Accuracy')
plt.title('AdaBoost - Decision Tree')
plt.show()