import numpy as np
import pandas as pd

##### Données ######

data_read = pd.read_csv("train.csv")

data = np.array(data_read)

np.random.shuffle(data)

X_dev = data[:1000, 1:] # Images de validation

Y_dev = data[:1000,0] # Classes de validation

X_train = data[1000:, 1:] # Images d'entraînement

Y_train = data[1000:,0] # Classes d'entraînement

# Normalisation :

maxi = X_train.max()
X_dev = X_dev/maxi
X_train = X_train/maxi


##### INITIALISATION #####

def initialise(npixel, ncouche, nsortie):
    """
    Initialisation des poids et des biais

    Args:
        npixel (array): _description_
        ncouche (int): nombre de neurones dans la couche cachée
        nsortie (int): nombre de neurones dans la couche de sortie

    Returns:
        W0 (int): poids des entrées à la couche cachée
        W1 (int): poids de la couche cachée à la sortie
        b0 (int): biais de la couche cachée
        b1 (int): biais de la couche de sortie
    """
    W0 = np.random.uniform(-0.5, 0.5, size=(npixel,ncouche))
    W1 = np.random.uniform(-0.5, 0.5, size=(ncouche,nsortie))
    b0 = np.random.uniform(-0.5, 0.5, ncouche)
    b1 = np.random.uniform(-0.5, 0.5, nsortie)

    return W0, W1, b0, b1

def relu(Z):
    """
    Fonction d'activation de la couche cachée

    Args:
        Z (array): variable de la fonction ReLu

    Returns:
        array: Fonction ReLu
    """
    return (Z > 0)*Z


def der_relu(Z):
    """
    Dérivée de la fonction d'activation de la couche cachée

    Args:
        Z (array): variable de la fonction ReLu

    Returns:
        array: Derivée fonction ReLu
    """
    return Z > 0

def softmax(Z):
    """
    Fonction d'activation de la couche de sortie

    Args:
        Z (array): variable de la fonction Softmax

    Returns:
        array: Fonction Softmax
    """
    sigma = np.exp(Z)/sum(np.exp(Z))

    return sigma


### Propagation avant ###

def prop_avant(data, poids):
    """
    Propagation avant d'une image

    Args:
        data (array): vecteur contenant les pixels d'une image
        poids (array): vecteur contenant les poids et biais [W0, W1, b0, b1]

    Returns:
        array:
        A0, A1 : vecteurs d'activation de la couche cachée/de sortie
        Z0, Z1 : vecteurs des arguments de la couche cachée/de sortie
    """
    Z0 = np.dot(np.transpose(poids[0]), data) + poids[2]
    A0 = relu(Z0)
    Z1 = np.dot(np.transpose(poids[1]), A0) + poids[3]
    A1 = softmax(Z1)

    return Z0, A0, Z1, A1

### Propagation arriere ###

def classe(label):
    """Ecris une classe sous forme binaire

    Args:
        label (int): classe, numéro entre 0 et 9

    Returns:
        array: vecteur de taille 10
    """
    Y = np.zeros(10)
    Y[label] = 1
    return Y

def prop_arriere(label, Z0, A0, Z1, A1, data, poids):
    """
    Propagation arrière pour 1 image

    Args:
        label (int): classe associée à l'image
        Z0 (array): vecteur des arguments de la couche cachée
        A0 (array): vecteur d'activation de la couche cachée
        Z1 (array): vecteurs des arguments de la couche de sortie
        A1 (array): vecteurs d'activation de la couche de sortie
        data (array): les pixels pour une seule image
        poids (array): poids et biais [W0, W1, b0, b1]

    Returns:
        array: # Je ne suis pas sûre, les 4 sont des vecteurs?
        dJdW0, dJdb0, dJdW1, dJdb1 : les différentes dérivées de la fonction de perte
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
    Actualise les poids, avec un taux d'apprentissage lambda_ donné

    Args:
        lambda_ (float): taux d'apprentissage
        dJdW0_L (list): list avec les différents dérivées de W0
        dJdb0_L (list): list avec les différents dérivées de b0
        dJdW1_L (lis): list avec les différents dérivées de w1
        dJdb1_L (list): list avec les différents dérivées de b1
        poids (array): poids et biais [W0, W1, b0, b1]

    Returns:
        float: nouveaux poids et biais
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
    Retourne la position de la probabilité maximum

    Args:
        probas (array): Vecteur des probabilités

    Returns:
        float: Position de la probabilité la plus haute
    """
    return np.argmax(probas)


def taux_succes(predictions, attendues):
    """
    Retourne le taux de succès d'images trouvées, pour une itération complète
    Le taux est normalisé à 1

    Args:
        predictions (array): chiffres predites
        attendues (array): chiffres attendues

    Returns:
        float: taux de succès
    """
    taux = np.sum(predictions==attendues)/np.size(predictions)
    return taux


def entrainement(X_train, Y_train, n_iterations, lambda_, poids=None, test = False):
    """
    Algorithme d'entrainement du réseau de neuronese

    Args:
        X_train (array): les images d'entraînement
        Y_train (array): les classes d'entraînement
        n_iterations (int): nombre d'itérations, époques
        lambda_ (float): taux d'apprentissage
        poids (array, optional): poids et biais initiaux. Defaults to None.
        test (bool, optional): permettant de réaliser ou pas l'analyse. Defaults to False.

    Returns:
        floats: poids et biais finaux
    """
    n_images = len(Y_train)         # correspond au nombre d'image sur lequel on s'entraine
    npixel = X_train[0].size   		# nombre de pixels dans 1 image
    ncouche = 10
    nsortie = 10
    attendu = Y_train # les valeurs d'images attendues ( pour les 41 000 images)
    taux = []
    if poids==None:
        # on initialise, même pour toutes les images
        W0, W1, b0, b1 = initialise(npixel, ncouche, nsortie)
        poids = [W0, W1, b0, b1]

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

        if test :
            taux.append(taux_succes(predictions, attendu))
            if i%10==0:
                print(taux_succes(predictions, attendu))
    if test:
        pathfiles = "files/"
        np.save(pathfiles + f"{int(lambda_)}taux", taux)
        print("File saved, path : " + pathfiles + f"{int(lambda_)}taux")
    return  W0, W1, b0, b1


def validation(X_dev, Y_dev, poids):
    """
    Retourne la classe prédite pour les images de l'échantillon de validation

    Args:
        X_dev (array): les images de validation
        Y_dev (array): les classes de validation
        poids (array): poids et biais

    Returns:
        float: taux de succès
    """
    n_images = len(Y_dev)
    predictions = []
    for i in range(n_images):
        xi = X_dev[i]
        Z0, A0, Z1, A1 = prop_avant(xi, poids)
        predictions.append(proba_max(A1))
    return taux_succes(predictions, Y_dev)