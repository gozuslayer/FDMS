%matplotlib inline
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



# chargement 
digits = load_digits()
x = digits.data

y = digits.target
# devision des data en train et test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)


def distance_all_points(X):

    N,M = X.shape
    X2 = np.sum(X**2,1);
    res_1 = np.zeros((N,N))
    res_2 = np.zeros((N,N))
    
    for i in range(N):
        res_1[i,:] = X2
    
    for i in range(N):
        res_2[:,i] = X2.T
    
    distance = res_1 + res_2 - 2*X.dot(X.T)
    
    return distance

def neighborhood (X_distance,k):

    tmp = np.argsort(X_distance)
    res = tmp[:,1:k+1]
    
    return res


from scipy.linalg import solve
from scipy.sparse import linalg, eye
import numpy as np 

def LLE (X,K):
    
    # calcule des weights 
    N,M = X.shape
    d = distance_all_points(X)
    n = neighborhood(d,K)
    W = np.zeros((N,K))
    point = np.zeros((K,M))
    voisins = np.zeros((K,M)) 
    WW = np.zeros((N,N))
    
    for ii in range(N):
        
        # le point original repeter k fois 
        for i in range(K):
            
            point[i,:] = X[ii,:] 
            
        # les voisins du point original     
        for j in range(K):

            kk = n[ii][j]
            voisins[j,:] = X[kk,:]
            
        z = point-voisins  
        A = z.dot(z.T)   
        B = np.ones(K)
        W[ii,:] = solve(A,B)       #  solve Cw=1
        W[ii,:] = W[ii,:]/np.sum(W[ii,:])    
        WW[ii][n[ii]] = W[ii]
        
    M = np.identity(N)-WW
    matrice = M.T.dot(M) 
    u,sigma,v = np.linalg.svd(matrice,full_matrices=0)
    indice = np.argsort(sigma)[1:4]
    Y = np.array(v[indice,:].T) 

    return Y

from scipy import stats
import numpy

class TSNE2():
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=1000.0, n_iter=1000, alpha=0):
        self.perplexity = perplexity # entre 5 et 50
        self.eta = learning_rate
        self.dim = n_components
        self.n_iter = n_iter
        self.alpha = alpha

    def fit_transform(self,x):
        self.x = x
        n,m = self.x.shape

        # Recherche dichotomique du sigma (entre le max des distances et le min)
        
        # Utiliser la décomposition matriciel pour calculer la norm ||xi-xj||² = <xi-xj; xi-xj> 
        #x_norm = np.sum(self.x**2,axis=1).reshape(1, n)
        x_norm = np.linalg.norm(self.x, axis =1).reshape(1,n)
        distancex = x_norm + x_norm.T - 2 * np.dot(self.x,self.x.T)
        
        inf = np.zeros((n,1))
        sup = np.ones((n,1)) * np.max(distancex)
        self.sigma = (inf+sup)/2.0
        
        self.lperp = np.log2(self.perplexity)


        #dichotomie        
        while True:
            #espace d'entré Pj/i
            self.pcond = np.exp( -distancex /(2.0*self.sigma**2) )
            self.pcond = self.pcond / (( self.pcond-np.eye(n) ).sum(axis=1) ).reshape( n,1 ) # ne pas compter si k=i,n

            self.H = - np.sum(  self.pcond * np.log2(self.pcond), axis=0 )
            self.perp = 2**self.H
            
            for i in xrange(n):
                if ( self.perp[i] < self.lperp ):
                    inf[i] = self.sigma[i]
                else:
                    sup[i] = self.sigma[i]
            sigma_old = self.sigma
            self.sigma = ((sup + inf) / 2.)
            if np.max(np.abs(sigma_old - self.sigma)) < 1e-5:
               break

        #Pij
        self.pij = (self.pcond+self.pcond.T) / (2.0*n)
        np.fill_diagonal(self.pij, 0)
        
        # init y
        self.y = np.zeros( (self.n_iter+2,n,self.dim) )
        self.y[1] =  numpy.random.normal(0, 1e-4, (n, self.dim))#1
        
        
        loss = []
        for t in xrange( 1, self.n_iter + 1 ):
            # qij
            y_norm = np.sum(self.y[t]**2,axis=1).reshape(1, n)
            distance_y = y_norm + y_norm.T - 2 * np.dot(self.y[t],self.y[t].T)
            
            #self.qij = 1 / ( 1 + distance_y )
            #np.fill_diagonal(self.qij, 0)
            #self.qij = self.qij/self.qij.sum()
            
            self.qij = 1 / ( 1 + distance_y )
            np.fill_diagonal(self.qij, 0)
            self.qij = (1 / ( 1 + distance_y ))/self.qij.sum()

            yt = self.y[t]
            grad = 4 * ( (self.pij-self.qij) / (1 + distance_y)).reshape(n,n,1)
            for i in range(n):
                grad_i = ( grad[i] * (yt[i] - yt) ).sum(0)
                self.y[t+1][i] = yt[i] - self.eta * grad_i + self.alpha * (yt[i] - self.y[t-1][i])            
            
            l = stats.entropy(self.pij, self.qij, 2).mean()
            loss.append(l)
            if (t % 1 == 30):
                print t,l, self.pij.mean(), self.qij.mean(), yt.mean()
        return self.y


def plot_LLE_3D (Y,Y_train):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    categories = Y_train
    colormap = np.array(['r', 'g', 'b','#16B84E','y','#87591A','c','m','#FF00FF','#D473D4'])
    t = ax.scatter(Y[:,0], Y[:,1] ,Y[:,2], s=80 , c=colormap[categories])
    plt.show()

def plot_LLE_2D (Y,Y_train):
    
    fig = plt.figure()
    categories = Y_train
    colormap = np.array(['r', 'g', 'b','#16B84E','y','#87591A','c','m','#FF00FF','#D473D4'])
    plt.scatter(Y[:,0], Y[:,1] , s=80 , c=colormap[categories])
    plt.show()

Y_LLE = LLE (X_train,17)

plot_LLE_3D (Y_LLE,Y_train)

plot_LLE_2D(Y_LLE,Y_train)

Y_t_SNE = TSNE2(n_components=2, perplexity=30.0, learning_rate=1000.0, n_iter=1000, alpha=0).fit_transform(X_train)

plot_LLE_2D(Y_t_SNE,Y_train)


# visualisation avec sklearn TSNE
from sklearn.manifold import TSNE

Y_t_SNE = TSNE(random_state=20150101).fit_transform(X_train)
plot_LLE_2D(Y_t_SNE,Y_train)