
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle as pkl

def loadMovieLens(path='D:/Users/Moi/Desktop/FDMS/ml-100k'):
    # Get movie titles
    movies={}
    for line in open(path+'/u.item'):
        (id,title)=line.split('|')[0:2]
        movies[id]=title

    # Load data
    prefs={}
    for line in open(path+'/u.data'):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]]=float(rating)
    return (prefs,movies.values())


    #transformation sous forme de triplet [id_user,movie,rate]
def getTriplet(data):
    triplet = []
    for u in data.keys():
        for i in data[u].keys():
            triplet.append([u,i,data[u][i]])
    return triplet

# Split l'ensemble des triplets  (80% train et 20% test )
def splitTrainTest(triplet, testProp) :
    perm = np.random.permutation(triplet)
    splitIndex = int(testProp * len(triplet))
    return perm[splitIndex:], perm[:splitIndex]

#constuction des dics a partir de triplet
#Pour simplifier la manipulation des données
def getDataByUsers(triplet) :
    dataByUsers = {}
    for t in triplet:
        if not t[0] in dataByUsers.keys():
            dataByUsers[t[0]] = {}
        dataByUsers[t[0]][t[1]] = float(t[2])
    return dataByUsers

def getDataByItems(triplet) :
    dataByItems = {}
    for t in triplet:
        if not t[1] in dataByItems.keys():
            dataByItems[t[1]] = {}
        dataByItems[t[1]][t[0]] = float(t[2])
    return dataByItems


# supprime des données de test => les données inconnus en train
def deleteUnknowData(triplet_test, trainUsers, trainItems) :
    to_Del = []
    for i,t in enumerate(triplet_test):
        if not t[0] in trainUsers:
            to_Del.append(i)
        elif not t[1] in trainItems:
            to_Del.append(i)
    return np.delete(triplet_test, to_Del, 0)

    # LoadMovieLense
data, movies = loadMovieLens()

triplet = getTriplet(data)
print(triplet[0])

# split 80% train 20% test
triplet_train, triplet_test = splitTrainTest(triplet , 0.2)
print(len(triplet_train))
print(len(triplet_test))

# getDataByUsers : {users : {items : note}}
# getDataByItems : {items : {users : note}} 

# train
trainUsers = getDataByUsers(triplet_train)
trainItems = getDataByItems(triplet_train)
print(trainUsers['1']['Pillow Book, The (1995)'])   


#aprés le split on teste et on garde que sur les films qui sont dans notre ensemble d'apprentissage
triplet_test = deleteUnknowData(triplet_test, trainUsers, trainItems)
print(len(triplet_test))


class FactoMatrice():
    def __init__(self, k, epsilon=1e-3, nbIter=2000, lamb=0.5):
        self.k = k
        self.lamb = lamb
        self.epsilon = epsilon
        self.nbIter = nbIter

        # descente de gradient stochastique avec mise à jour altérnée
    def fit(self, trainUsers, triplet):
        self.p = {}
        self.q = {}
        self.triplet = triplet
        for j in range(len(triplet)): # On initialise les cases vides en random
                u = triplet[j][0]
                i = triplet[j][1]
                if not u in self.p:
                    self.p[u] = np.random.rand(1,self.k) # matrice ligne pour un users
                if not i in self.q:
                    self.q[i] = np.random.rand(self.k,1) # matrice colonne pour un item
        loss = []     
        for it in range(self.nbIter):
            ind = np.random.randint(len(triplet))
            u = triplet[ind][0]
            i = triplet[ind][1]
            
            tmp = trainUsers[u][i] - self.p[u].dot(self.q[i])[0][0]
            self.p[u] = (1 - self.lamb * self.epsilon) * self.p[u] + self.epsilon * 2 * tmp * self.q[i].transpose()
            self.q[i] = (1 - self.lamb * self.epsilon) * self.q[i] + self.epsilon * 2 * tmp * self.p[u].transpose()
            
            loss.append(tmp*tmp) 
            
            if ((it)%(self.nbIter*0.2) == 0) :
                print 'itération : {}'.format(it)
                print "loss: {}".format(np.mean(loss))
                print "-------"
                loss = []
                
    def predict(self, triplet_test):
        pred = np.zeros(len(triplet_test))
        for ind,t in enumerate(triplet_test):
            pred[ind] = self.p[t[0]].dot(self.q[t[1]])[0][0]
        return pred
    
    def score(self, triplet_test) :
        return ((self.predict(triplet_test) - np.array(triplet_test[:,2], float)) ** 2).mean()

        k = 5
epsilon = 7e-3
nbIter = 25*len(triplet_train)
lamb = 0.2
model = FactoMatrice(k, epsilon=epsilon, nbIter=nbIter,lamb=lamb)
model.fit(trainUsers, triplet_train)
print "erreur en test:", model.score(triplet_test)

class FactoMatriceBiais():
    def __init__(self, k, epsilon=1e-3, nbIter=2000, lamb=0.5):
        self.k = k
        self.lamb = lamb
        self.epsilon = epsilon
        self.nbIter = nbIter

    def fit(self, trainUsers, triplet):

        self.p = {}
        self.q = {}
        self.bu = {} #biais sur les utilisateurs
        self.bi = {} #biais sur les items
        self.mu = np.random.random() * 2 - 1
        
        for j in range(len(triplet)): # On initialise les cases vides en random
            u = triplet[j][0]
            i = triplet[j][1]
            if not u in self.p:
                self.p[u] = np.random.rand(1,self.k) # matrice ligne pour un users
                self.bu[u] = np.random.rand() * 2 - 1
            if not i in self.q:
                self.q[i] = np.random.rand(self.k,1) # matrice colonne pour un item
                self.bi[i] = np.random.rand() * 2 - 1
        loss = []   
        for it in range(self.nbIter):
            ind = np.random.randint(len(triplet))
            u = triplet[ind][0]
            i = triplet[ind][1]
            
            tmp = trainUsers[u][i] - (self.mu + self.bi[i] + self.bu[u] +self.p[u].dot(self.q[i])[0][0])
            self.p[u] = (1 - self.lamb * self.epsilon) * self.p[u] + self.epsilon * 2 * tmp * self.q[i].transpose()
            self.bu[u] = (1 - self.lamb * self.epsilon) * self.bu[u] + self.epsilon * 2 * tmp
            self.q[i] = (1 - self.lamb * self.epsilon) * self.q[i] + self.epsilon * 2 * tmp * self.p[u].transpose()
            self.bi[i] = (1 - self.lamb * self.epsilon) * self.bi[i] + self.epsilon * 2 * tmp
            self.mu = (1 - self.lamb * self.epsilon) * self.mu + self.epsilon * 2 * tmp
            
            loss.append(tmp*tmp) 
            if ((it)%(self.nbIter*0.2) == 0) :
                print "itération : " , it
                print "loss : ", np.mean(loss)
                print "-------"
                loss = []
            
    def predict(self, triplet_test):
        pred = np.zeros(len(triplet_test))
        for ind,t in enumerate(triplet_test):
            pred[ind] = self.mu + self.bu[t[0]] + self.bi[t[1]] + self.p[t[0]].dot(self.q[t[1]])[0][0]
        return pred
    
    def score(self, triplet_test) :
        return ((self.predict(triplet_test) - np.array(triplet_test[:,2], float)) ** 2).mean()

        
k = 5
epsilon = 7e-3
nbIter = 25*len(triplet_train)
lamb = 0.2
model = FactoMatriceBiais(k, epsilon=epsilon, nbIter=nbIter,lamb=lamb)
model.fit(trainUsers, triplet_train)
print "erreur en test:", model.score(triplet_test)