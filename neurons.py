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
	
def prop_avant(data, poids):
	"""
	data : vecteur data pour une seule image, on va le faire pour chaque image
	poids = poids et biais [W0, W1, b0, b1]
	"""

	Z0 = np.dot(np.transpose(poids[0]), data) + poids[2]
	A0 = relu(Z0)
	Z1 = np.dot(np.transpose(poids[1]), A0) + poids[3]
	A1 = softmax(Z1)
	
	return Z0, A0, Z1, A1
	
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



def actualisation(label, Z0, A0, Z1, A1, data, poids):
	
	dJdW0, dJdb0, dJdW1, dJdb1 = prop_arriere(label, Z0, A0, Z1, A1, data, poids)
	dJdW0_L = [dJdW0]
	dJdb0_L = [dJdb0]
	dJdW1_L = [dJdW1]
	dJdb1_L = [dJdb1]
	
	
	return W0, W1, b0, b1



	
