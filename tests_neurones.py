import neurons as n
import numpy as np
import matplotlib.pyplot as plt

import style

path2plot = "plots/"
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
print("Shape matrice poids W0", np.shape(W0), " elle doit être (784, 10)")
print("Shape matrice poids W1", np.shape(W1), " elle doit être (10, 10)")
print("Shape matrice biais b0", np.shape(b0), " elle doit être (10,)")
print("Shape matrice biais b1", np.shape(b1), " elle doit être (10,)")

Z = np.arange(-10,10,1)

## RELU

fig, ax = plt.subplots()
ax.plot(Z, n.relu(Z), c='b', ls='-', label="relu")

ax.set_xlabel("Z")
ax.set_ylabel("relu")
#ax.set_title("Profil amas sans structure qui fusionne")
plotnom = path2plot + 'relu.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

## Derivée RELU

fig, ax = plt.subplots()
ax.plot(Z, n.der_relu(Z), c='b', ls='-', label="der_relu")

ax.set_xlabel("Z")
ax.set_ylabel("der_relu")
#ax.set_title("Profil amas sans structure qui fusionne")
plotnom = path2plot + 'der_relu.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

## SIGMA

fig, ax = plt.subplots()
ax.plot(Z, n.softmax(Z), c='b', ls='-', label="softmax")

ax.set_xlabel("Z")
ax.set_ylabel("softmax")
#ax.set_title("Profil amas sans structure qui fusionne")
plotnom = path2plot + 'softmax.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

## Propagation avant
poids = [W0, W1, b0, b1]
Z0, A0, Z1, A1 = n.prop_avant(n.X_dev[0], poids) # X_dev[0] pour tester juste la première image.
print("Shape matrice Z0", np.shape(Z0), " elle doit être (10,)")
print("Shape matrice A0", np.shape(A0), " elle doit être (10,)")
print("Shape matrice Z1", np.shape(Z1), " elle doit être (10,)")
print("Shape matrice A1", np.shape(A1), " elle doit être (10,)")

## Test Classe:

label = 7
print("Classe correspondant au chiffre", label, "est :", n.classe(7))

## Propagation arriere

dJdW0, dJdb0, dJdW1, dJdb1 = n.prop_arriere(label, Z0, A0, Z1, A1, n.X_dev[0], poids) # X_dev[0] pour tester juste la première image.
print("Shape matrice dJdW0", np.shape(dJdW0), " elle doit être (784,10)")
print("Shape matrice dJdb0", np.shape(dJdb0), " elle doit être (10,)")
print("Shape matrice dJdW1", np.shape(dJdW1), " elle doit être (10,10)")
print("Shape matrice dJdb1", np.shape(dJdb1), " elle doit être (10,)")






# Test de taux succes:

a = np.array([1,2,3,4])
b = np.array([1,3,5,4])
print("taux attendu: 0.5, taux de succes trouvé:", n.taux_succes(a,b))

a = np.array([1,2,3,4])
b = np.array([1,2,5,4])
print("taux attendu: 0.75, taux de succes trouvé:", n.taux_succes(a,b))


# Test de l'entrainement:

n_iterations = 100
lambda_ = 1.0


W0, W1, b0, b1 = n.entrainement(n.X_train, n.Y_train, n_iterations, lambda_)

np.save("W0", W0)
np.save("W1", W1)
np.save("b0", b0)
np.save("b1", b1)

lambda_ = 0.1
poids = [W0, W1, b0, b1]

W0, W1, b0, b1 = n.entrainement(n.X_train, n.Y_train, n_iterations, lambda_, poids)

np.save("W0", W0)
np.save("W1", W1)
np.save("b0", b0)
np.save("b1", b1)













