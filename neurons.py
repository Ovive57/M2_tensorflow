import numpy as np
import pandas as pd

##### Échantillons d'entraînement ######

data_read = pd.read_csv("train.csv")

data = np.array(data_read)

np.random.shuffle(data)

X_dev = data[:1000, 1:] # 1000 premieres images du tableau avec les classes

Y_dev = data[:1000,0] # 1000 premieres classes du tableau associées

X_train = data[1000:, 1:] # Images et classes d'entraînement

Y_train = data[1000:,0] # Classes d'entraînement

# Normalisation : 

maxi = X_train.max()
X_dev = X_dev/maxi
X_train = X_train/maxi


##### INITIALISATION #####

def initialise(npixel, ncouche, nsortie):
	"""
	npixel = np.shape(X_dev[0]) # 784 pixels
	ncouche = 10
	nsortie = 10
	"""

	W0 = np.random.uniform(-0.5, 0.5, size=(npixel,ncouche))
	W1 = np.random.uniform(-0.5, 0.5, size=(ncouche,nsortie))
	b0 = np.random.uniform(-0.5, 0.5, ncouche)
	b1 = np.random.uniform(-0.5, 0.5, nsortie)
	
	return W0, W1, b0, b1
	
	
def relu(Z):
	
	return (Z > 0)*Z
	
	
def der_relu(Z):
	return Z > 0
	
def softmax(Z):
	
	sigma = np.exp(Z)/sum(np.exp(Z))
	
	return sigma

### Propagation avant ###


def prop_avant(data, poids):
	"""
	data : vecteur data (pixels) pour une seule image, on va le faire pour chaque image
	poids = poids et biais [W0, W1, b0, b1]
	"""

	Z0 = np.dot(np.transpose(poids[0]), data) + poids[2]
	A0 = relu(Z0)
	Z1 = np.dot(np.transpose(poids[1]), A0) + poids[3]
	A1 = softmax(Z1)
	
	return Z0, A0, Z1, A1
	
### Propagation arriere ###

def classe(label):

	Y = np.zeros(10)
	Y[label] = 1
	
	return Y
	
def prop_arriere(label, Z0, A0, Z1, A1, data, poids):
	
	"""
	data : vecteur data (les pixels) pour une seule image, on va le faire pour chaque image
	poids = poids et biais [W0, W1, b0, b1]
	"""
	Y = classe(label)
	
	delta1 = A1 - Y
	
	delta0 = der_relu(Z0) * np.dot(poids[1],delta1)
	
	dJdW0 = np.dot(data[:,None], delta0[None,:])
	dJdW1 = np.dot(A0[:,None], delta1[None,:])
	
	dJdb0 = delta0
	dJdb1 = delta1
	
	return dJdW0, dJdb0, dJdW1, dJdb1



def actualisation(lambda_, dJdW0_L, dJdb0_L, dJdW1_L, dJdb1_L, poids):
    """
    
    """
    ncouples = len(dJdW0_L) # à tester
    
    # On actualise les poids et biais
    W1 = poids[1] - lambda_/ncouples*np.sum(dJdW1_L, axis = 0)
    W0 = poids[0] - lambda_/ncouples*np.sum(dJdW0_L, axis = 0)

    b1 = poids[3] - lambda_/ncouples*np.sum(dJdb1_L, axis = 0)
    b0 = poids[2] - lambda_/ncouples*np.sum(dJdb0_L, axis = 0)

    return W0, W1, b0, b1


def proba_max(probas):
    """
     probas les probabilités en sorties 
    """
    return np.argmax(probas)
        
        
def taux_succes(predictions, attendues):
    taux = np.sum(predictions==attendues)/np.size(predictions)
    return taux


def entrainement(X_train, Y_train, n_iterations):
    n_images = len(Y_train)         # correspond au nombre d'image sur lequel on s'entraine
    npixel = X_train[0].size   # nombre de pixels dans 1 image
    ncouche = 10
    nsortie = 10
    attendu = Y_train # le valeurs d'image attendu ( pour les 41 000 images)
    # on initialise, même pour toutes les images
    W0, W1, b0, b1 = initialise(npixel, ncouche, nsortie)
    poids = [W0, W1, b0, b1]
    lambda_ = 1.0
    for i in range(n_iterations):
        dJdW0_L = []
        dJdb0_L = []
        dJdW1_L = []
        dJdb1_L = []
        predictions = [] # le vecteur avec toutes les valeurs trouvées
        for j in range(n_images):
            xi = X_train[j]   # pixels de l'image actuelle
            yi = Y_train[j]   # label de l'image actuelle ( taille 1 )
            Z0, A0, Z1, A1 = prop_avant(xi, poids)
            dJdW0, dJdb0, dJdW1, dJdb1 = prop_arriere(yi, Z0, A0, Z1, A1, xi, poids)
            dJdW0_L.append(dJdW0)
            dJdb0_L.append(dJdb0)
            dJdW1_L.append(dJdW1)
            dJdb1_L.append(dJdb1)
            predictions.append(proba_max(A1)) # le chiffre trouvé pour l'image actuelle
        W0, W1, b0, b1 = actualisation(lambda_, dJdW0_L, dJdb0_L, dJdW1_L, dJdb1_L, poids) # avec les nouvelles listes de dJ, on actualise les poids
        poids = [W0, W1, b0, b1]
        if i%10==0:
            print(taux_succes(predictions, attendu))
    return  W0, W1, b0, b1
    

"""
à faire ensuite:
en argument d'entrainement, mettre:
W0, W1, b0, b1 = None
et on les initialise direct avec ceux trouvés pour aller plus vite
"""












