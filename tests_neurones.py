import neurons as n
import numpy as np
import matplotlib.pyplot as plt
import style

path2plot = "plots/"
pathfiles = "files/"
plt.style.use(style.style1)


print("Dimension de data : ", np.shape(n.data)) # 785, premier indice c'est la classe et les 784 autres sont les pixels de chaque image. 42000 images

print("On prend", len(n.X_dev)," images pour le test.")
print("On prend", len(n.X_train), " images pour l'entrainement.")

print("Shapes de : X_train:", np.shape(n.X_train), ", de Y_train:", np.shape(n.Y_train))

### TEST FONCTIONS INITIALISATION ###

npixel = n.X_dev[0].size # 784 pixels
ncouche = 10
nsortie = 10

W0, W1, b0, b1 = n.initialise(npixel, ncouche, nsortie)
print("\nTEST FONCTION INITIALISATION")
print("Shape matrice poids W0", np.shape(W0), " elle doit être (784, 10)")
print("Shape matrice poids W1", np.shape(W1), " elle doit être (10, 10)")
print("Shape matrice biais b0", np.shape(b0), " elle doit être (10,)")
print("Shape matrice biais b1", np.shape(b1), " elle doit être (10,)")

Z = np.arange(-10,10,1)

# Test RELU

fig, ax = plt.subplots()
ax.plot(Z, n.relu(Z), c='b', ls='-')
ax.set_xlabel("Z")
ax.set_ylabel("relu")
ax.set_title("Fonction ReLu")
plotnom = path2plot + 'relu.pdf'
plt.savefig(plotnom)


# Test Derivée RELU

fig, ax = plt.subplots()
ax.plot(Z, n.der_relu(Z), c='b', ls='-')

ax.set_xlabel("Z")
ax.set_ylabel("der_relu")
ax.set_title("Derivée fonction ReLu")
plotnom = path2plot + 'der_relu.pdf'
plt.savefig(plotnom)


# Test Softmax

fig, ax = plt.subplots()
ax.plot(Z, n.softmax(Z), c='b', ls='-')

ax.set_xlabel("Z")
ax.set_ylabel("softmax")
ax.set_title("Fonction Softmax")
plotnom = path2plot + 'softmax.pdf'
ax.legend()
plt.savefig(plotnom)


### TEST FONCTIONS PROPAGATION AVANT ###

poids = [W0, W1, b0, b1]
Z0, A0, Z1, A1 = n.prop_avant(n.X_dev[0], poids) # X_dev[0] pour tester juste la première image.
print("\nTEST FONCTION PROPAGATION AVANT")
print("Shape matrice Z0", np.shape(Z0), " elle doit être (10,)")
print("Shape matrice A0", np.shape(A0), " elle doit être (10,)")
print("Shape matrice Z1", np.shape(Z1), " elle doit être (10,)")
print("Shape matrice A1", np.shape(A1), " elle doit être (10,)")

# Test Classe:

label = 7
print("\nTEST FONCTION CLASSE")
print("Classe correspondant au chiffre", label, "est :", n.classe(7))

### TEST FONCTIONS PROPAGATION ARRIÈRE ###

dJdW0, dJdb0, dJdW1, dJdb1 = n.prop_arriere(label, Z0, A0, Z1, A1, n.X_dev[0], poids) # X_dev[0] pour tester juste la première image.
print("\nTEST PROPAGATION ARRIÈRE")
print("Shape matrice dJdW0", np.shape(dJdW0), " elle doit être (784,10)")
print("Shape matrice dJdb0", np.shape(dJdb0), " elle doit être (10,)")
print("Shape matrice dJdW1", np.shape(dJdW1), " elle doit être (10,10)")
print("Shape matrice dJdb1", np.shape(dJdb1), " elle doit être (10,)")


# Test de la fonction taux de succes:

a = np.array([1,2,3,4])
b = np.array([1,3,5,4])
print("\nTEST FONCTION TAUX DE SUCCES")
print("taux attendu: 0.5, taux de succes trouvé:", n.taux_succes(a,b))

a = np.array([1,2,3,4])
b = np.array([1,2,5,4])
print("taux attendu: 0.75, taux de succes trouvé:", n.taux_succes(a,b))


### ENTRAINEMENT ###

# Premier entrainement avec lambda = 1. :
print("\nPREMIER ENTRAINEMENT")
n_iterations = 100
lambda_ = 1.0

W0, W1, b0, b1 = n.entrainement(n.X_train, n.Y_train, n_iterations, lambda_, test = True)

np.save(pathfiles + "W0", W0)
np.save(pathfiles + "W1", W1)
np.save(pathfiles + "b0", b0)
np.save(pathfiles + "b1", b1)


# Second entrainement avec lambda = 0.1 :
print("\nSECOND ENTRAINEMENT")
W0 = np.load(pathfiles + "W0.npy")
W1 = np.load(pathfiles + "W1.npy")
b0 = np.load(pathfiles + "b0.npy")
b1 = np.load(pathfiles + "b1.npy")

n_iterations = 100
lambda_ = 0.1

poids = [W0, W1, b0, b1]

W0_01, W1_01, b0_01, b1_01 = n.entrainement(n.X_train, n.Y_train, n_iterations, lambda_, poids, test = True)

np.save(pathfiles + "W0_01", W0_01)
np.save(pathfiles + "W1_01", W1_01)
np.save(pathfiles + "b0_01", b0_01)
np.save(pathfiles + "b1_01", b1_01)


# Calcule des taux de succès de l'échantillon de validation :

# Pour le premier entrainement:

W0 = np.load(pathfiles + "W0.npy")
W1 = np.load(pathfiles + "W1.npy")
b0 = np.load(pathfiles + "b0.npy")
b1 = np.load(pathfiles + "b1.npy")
poids = [W0, W1, b0, b1]

taux_succes = n.validation(n.X_dev, n.Y_dev, poids)
print("\nTAUX PREMIER ENTRAINEMENT")
print("Le taux de succès obtenus après un entrainement de 100 itérations à lambda = 1:", taux_succes)

# Pour le second:

W0_01 = np.load(pathfiles + "W0_01.npy")
W1_01 = np.load(pathfiles + "W1_01.npy")
b0_01 = np.load(pathfiles + "b0_01.npy")
b1_01 = np.load(pathfiles + "b1_01.npy")
poids_01 = [W0_01, W1_01, b0_01, b1_01]

taux_succes_01 = n.validation(n.X_dev, n.Y_dev, poids_01)
print("\nTAUX SECOND ENTRAINEMENT")
print("Le taux de succès obtenus après un second entrainement de 100 itérations à lambda=0.1:", taux_succes_01)


# Plot de l'évolution du taux de succès pour les deux taux d'apprentissage

taux0 = np.load(pathfiles + "0taux.npy")
taux1 = np.load(pathfiles + "1taux.npy")
conv = np.ones(len(taux0))

fig, ax = plt.subplots()
ax.plot(conv, c = 'k', ls = '--', label = "convergence")
ax.plot(taux1, c='b', ls='-', label="1.0")
ax.plot(taux0, c='r', ls='-', label="0.1")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès pour deux taux d'apprentissage")
plotnom = path2plot + 'taux_succes_test.pdf'
ax.legend()
plt.savefig(plotnom)