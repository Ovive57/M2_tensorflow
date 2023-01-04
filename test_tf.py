import numpy as np
import matplotlib.pyplot as plt
import style

path2plot = "plots/"
pathfiles = "files/"
plt.style.use(style.style1)


#### MODELE 1 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones. 
# Taux d'apprentissage de 0.01.
# 300 époques.
# batch_size 60000.

# Ouvre le fichier contenant le dictionaire out pour le premier modèle:
out1 = np.load(pathfiles + 'out1.npy', allow_pickle='TRUE').item()


print("Clés associées au dictionnaire out.history, pour tous les modeles sont :\n", out1.keys(),
"\noù loss et val_loss sont la fonction de perte pour l'entraînement et le test respectivement,",
"\net accuracy et val_accuracy le taux de succès pour l'entraînement et le test respectivement.")


""" Olivia : J'enleverais ça, sauf si tu veux pas, je te laisse choisir !
print("Fonction perte entraînement : ", len(out1['loss']))
print("Taux de succès entraînement : ", len(out1['accuracy']))
print("Fonction perte test : ", len(out1['val_loss']))
print("Taux de succès test : ", len(out1['val_accuracy']))
print("Taille des listes des 4 entrées du dict: 300 (= au nombre d'époques)")
"""

print("\nMODELE 1 : lambda = 0.01, 300 époques.\n")

# Epoques
iterations = np.arange(len(out1['loss']))

# Taux de succès
taux_entr = np.max(out1['accuracy'])
taux_val = np.max(out1['val_accuracy'])
taux_max = np.max([out1['accuracy'], out1['val_accuracy']])

# Convergences
conv_perte = np.zeros(len(out1['loss']))
conv_perte_test = np.ones(len(out1['val_loss']))*np.min(out1['val_loss'])
conv_entr = np.ones(len(out1['loss']))*taux_entr
conv_val = np.ones(len(out1['loss']))*taux_val

# Plot Function de perte
fig, ax = plt.subplots()
ax.plot(iterations, conv_perte, c = 'k', ls = '--', label = "convergence souhaitée")
ax.plot(iterations,conv_perte_test, c = 'r', ls = '--', label = "convergence test")
ax.plot(out1['loss'], c='b', ls='-', label="train")
ax.plot(out1['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_1.pdf'
ax.legend()
plt.savefig(plotnom)

# Plot Taux de succès
fig, ax = plt.subplots()
ax.plot(iterations,conv_entr, c = 'b', ls = '--', label = "train convergence")
ax.plot(iterations,conv_val, c = 'r', ls = '--', label = "test convergence")
ax.plot(out1['accuracy'], c='b', ls='-', label="train")
ax.plot(out1['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_1.pdf'
ax.legend()
plt.savefig(plotnom)

# Taux de succès
print("Pour le modèle 1, avec", len(out1['accuracy']), " epochs, le taux de succès maximal est de :", taux_val)
print("La dernière valeur est :", out1['val_accuracy'][-1])


#### MODELE 2 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 1000 époques, pour essayer qu'il converge à 10⁻3 près.
# batch_size 60000.

print("\nMODELE 2 : lambda = 0.01, 1000 époques.\n")

out2=np.load(pathfiles + 'out2.npy',allow_pickle='TRUE').item()

# Epoques
iterations = np.arange(len(out2['loss']))

# Taux de succès
taux_entr_2 = np.max(out2['accuracy'])
taux_val_2 = np.max(out2['val_accuracy'])
taux_max_2 = np.max([out2['accuracy'], out2['val_accuracy']])

# Convergences
conv_perte_2 = np.zeros(len(out2['loss']))
conv_perte_test_2 = np.ones(len(out2['val_loss']))*np.min(out2['val_loss'])
conv_entr_2 = np.ones(len(out2['loss']))*taux_entr_2
conv_val_2 = np.ones(len(out2['loss']))*taux_val_2

# Plot Function de perte
fig, ax = plt.subplots()
ax.plot(iterations,conv_perte_2, c = 'k', ls = '--', label = "convergence souhaitée")
ax.plot(iterations,conv_perte_test_2, c = 'r', ls = '--', label = "convergence test")
ax.plot(out2['loss'], c='b', ls='-', label="train")
ax.plot(out2['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_2.pdf'
ax.legend()
plt.savefig(plotnom)

# Plot Taux de succès
fig, ax = plt.subplots()
ax.plot(iterations,conv_entr_2, c = 'b', ls = '--', label = "train convergence")
ax.plot(iterations,conv_val_2, c = 'r', ls = '--', label = "test convergence")
ax.plot(iterations, out2['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out2['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_2.pdf'
ax.legend()
plt.savefig(plotnom)

# Taux de succès
print("Pour le modèle 2, avec", len(out2['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_2)
print("La dernière valeur est :", out2['val_accuracy'][-1])


#### MODELE 2 bis ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 10000 époques, pour essayer qu'il converge à 10⁻3 près où voir le surentraînement.
# batch_size 60000.

print("\nMODELE 2 bis: lambda = 0.01, 10000 époques.\n")

out2b=np.load(pathfiles + 'out2b.npy',allow_pickle='TRUE').item()

# Epoques
iterations = np.arange(len(out2b['loss']))

# Taux de succès
taux_entr_2b = np.max(out2b['accuracy'])
taux_val_2b = np.max(out2b['val_accuracy'])
taux_max_2b = np.max([out2b['accuracy'], out2b['val_accuracy']])

# Convergences
conv_perte_2b = np.zeros(len(out2b['loss']))
conv_perte_test_2b = np.ones(len(out2b['val_loss']))*np.min(out2b['val_loss'])
conv_entr_2b = np.ones(len(out2b['loss']))*taux_entr_2b
conv_val_2b = np.ones(len(out2b['loss']))*taux_val_2b

# Plot Function de perte
fig, ax = plt.subplots()
ax.plot(iterations,conv_perte_2b, c = 'k', ls = '--', label = "convergence souhaitée")
ax.plot(iterations,conv_perte_test_2b, c = 'r', ls = '--', label = "convergence test")
ax.plot(out2b['loss'], c='b', ls='-', label="train")
ax.plot(out2b['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_2b.pdf'
ax.legend()
plt.savefig(plotnom)

# Plot Taux de succès
fig, ax = plt.subplots()
ax.plot(iterations,conv_entr_2b, c = 'b', ls = '--', label = "train convergence")
ax.plot(iterations,conv_val_2b, c = 'r', ls = '--', label = "test convergence")
ax.plot(iterations, out2b['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out2b['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_2b.pdf'
ax.legend()
plt.savefig(plotnom)

# Taux de succès
print("Pour le modèle 2 bis, avec", len(out2b['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_2b)
print("La dernière valeur est :", out2b['val_accuracy'][-1])


#### MODELE 2 bisbis ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 500 époques, pour essayer qu'il converge à 10⁻3 près où voir le surentraînement.
# batch_size 60000.

print("\nMODELE 2 bisbis: lambda = 0.01, 500 époques.\n")

out2bb=np.load(pathfiles + 'out2bb.npy',allow_pickle='TRUE').item()

# Epoques
iterations = np.arange(len(out2bb['loss']))

# Taux de succès
taux_entr_2bb = np.max(out2bb['accuracy'])
taux_val_2bb = np.max(out2bb['val_accuracy'])
taux_max_2bb = np.max([out2bb['accuracy'], out2bb['val_accuracy']])

# Convergences
conv_perte_2bb = np.zeros(len(out2bb['loss']))
conv_perte_test_2bb = np.ones(len(out2bb['val_loss']))*np.min(out2bb['val_loss'])
conv_entr_2bb = np.ones(len(out2bb['loss']))*taux_entr_2bb
conv_val_2bb = np.ones(len(out2bb['loss']))*taux_val_2bb

# Plot Function de perte
fig, ax = plt.subplots()
ax.plot(iterations,conv_perte_2bb, c = 'k', ls = '--', label = "convergence souhaitée")
ax.plot(iterations,conv_perte_test_2bb, c = 'r', ls = '--', label = "convergence test")
ax.plot(out2bb['loss'], c='b', ls='-', label="train")
ax.plot(out2bb['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_2bb.pdf'
ax.legend()
plt.savefig(plotnom)

# Plot Taux de succès
fig, ax = plt.subplots()
ax.plot(iterations,conv_entr_2bb, c = 'b', ls = '--', label = "train convergence")
ax.plot(iterations,conv_val_2bb, c = 'r', ls = '--', label = "test convergence")
ax.plot(iterations, out2bb['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out2bb['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_2bb.pdf'
ax.legend()
plt.savefig(plotnom)

# Taux de succès
print("Pour le modèle 2 bis bis, avec", len(out2bb['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_2bb)
print("La dernière valeur est :", out2bb['val_accuracy'][-1])


#### MODELE 3 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.2.
# 300 époques.
# batch_size 60000.

print("\nMODELE 3 : lambda = 0.2, 300 époques.\n")

out3=np.load(pathfiles + 'out3.npy', allow_pickle='TRUE').item()

# Epoques
iterations = np.arange(len(out3['loss']))

# Taux de succès
taux_entr_3 = np.max(out3['accuracy'])
taux_val_3 = np.max(out3['val_accuracy'])
taux_max_3 = np.max([out3['accuracy'], out3['val_accuracy']])

# Convergences
conv_perte_3 = np.zeros(len(out3['loss']))

# Plot Function de perte
fig, ax = plt.subplots()
ax.plot(iterations,conv_perte_3, c = 'k', ls = '--', label = "convergence souhaitée")
ax.plot(out3['loss'], c='b', ls='-', label="train")
ax.plot(out3['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_3.pdf'
ax.legend()
plt.savefig(plotnom)

# Plot Taux de succès
fig, ax = plt.subplots()
ax.plot(iterations, out3['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out3['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_3.pdf'
ax.legend()
plt.savefig(plotnom)

# Taux de succès
print("Pour le modèle 3, avec", len(out3['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_3)
print("La dernière valeur est :", out3['val_accuracy'][-1])


#### MODELE 4 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10*50 = 500 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 300 époques.
# batch_size 60000.

print("\nMODELE 4 : lambda = 0.01, 300 époques.\n")

out4=np.load(pathfiles + 'out4.npy', allow_pickle='TRUE').item()

# Epoques
iterations = np.arange(len(out4['loss']))

# Taux de succès
taux_entr_4 = np.max(out4['accuracy'])
taux_val_4 = np.max(out4['val_accuracy'])
taux_max_4 = np.max([out4['accuracy'], out4['val_accuracy']])

# Convergences
conv_perte_4 = np.zeros(len(out4['loss']))
conv_perte_test_4 = np.ones(len(out4['val_loss']))*np.min(out4['val_loss'])
conv_entr_4 = np.ones(len(out4['loss']))*taux_entr_4
conv_val_4 = np.ones(len(out4['loss']))*taux_val_4

# Plot Function de perte
fig, ax = plt.subplots()
ax.plot(iterations,conv_perte_4, c = 'k', ls = '--', label = "convergence souhaitée")
ax.plot(iterations,conv_perte_test_4, c = 'r', ls = '--', label = "convergence test")
ax.plot(out4['loss'], c='b', ls='-', label="train")
ax.plot(out4['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_4.pdf'
ax.legend()
plt.savefig(plotnom)

# Plot Taux de succès
fig, ax = plt.subplots()
ax.plot(iterations,conv_entr_4, c = 'b', ls = '--', label = "train convergence")
ax.plot(iterations,conv_val_4, c = 'r', ls = '--', label = "test convergence")
ax.plot(iterations, out4['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out4['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_4.pdf'
ax.legend()
plt.savefig(plotnom)

# Taux de succès
print("Pour le modèle 4, avec", len(out4['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_4)
print("La dernière valeur est :", out4['val_accuracy'][-1])


#### MODELE 5 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10*50 = 500 neurones, une deuxième couche cachée de 700 neurones,
# une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 200 époques.
# batch_size 60000.

print("\nMODELE 5 : lambda = 0.01, 200 époques.\n")

out5=np.load(pathfiles + 'out5.npy',allow_pickle='TRUE').item()

# Epoques
iterations = np.arange(len(out5['loss']))

# Taux de succès
taux_entr_5 = np.max(out5['accuracy'])
taux_val_5 = np.max(out5['val_accuracy'])
taux_max_5 = np.max([out5['accuracy'], out5['val_accuracy']])

# Convergences
conv_perte_5 = np.zeros(len(out5['loss']))
conv_perte_test_5 = np.ones(len(out5['val_loss']))*np.min(out5['val_loss'])
conv_entr_5 = np.ones(len(out5['loss']))*taux_entr_5
conv_val_5 = np.ones(len(out5['loss']))*taux_val_5

# Plot Function de perte
fig, ax = plt.subplots()
ax.plot(iterations,conv_perte_5, c = 'k', ls = '--', label = "convergence souhaitée")
ax.plot(iterations,conv_perte_test_5, c = 'r', ls = '--', label = "convergence test")
ax.plot(out5['loss'], c='b', ls='-', label="train")
ax.plot(out5['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_5.pdf'
ax.legend()
plt.savefig(plotnom)

# Plot Taux de succès
fig, ax = plt.subplots()
ax.plot(iterations,conv_entr_5, c = 'b', ls = '--', label = "train convergence")
ax.plot(iterations,conv_val_5, c = 'r', ls = '--', label = "test convergence")
ax.plot(iterations, out5['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out5['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du tauc de succès")
plotnom = path2plot + 'taux_succes_5.pdf'
ax.legend()
plt.savefig(plotnom)

# Taux de succès
print("Pour le modèle 5, avec", len(out5['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_5)
print("La dernière valeur est :", out5['val_accuracy'][-1])


#### MODELE 6 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10*50 = 500 neurones, une deuxième couche cachée de 700 neurones,
# une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 200 époques.
# batch_size 60000/10 = 6000.

print("\nMODELE 6 : lambda = 0.01, 200 époques.\n")

out6=np.load(pathfiles + 'out6.npy',allow_pickle='TRUE').item()

# Epoques
iterations = np.arange(len(out6['loss']))

# Taux de succès
taux_entr_6 = np.max(out6['accuracy'])
taux_val_6 = np.max(out6['val_accuracy'])
taux_max_6 = np.max([out6['accuracy'], out6['val_accuracy']])

# Convergences
conv_perte_6 = np.zeros(len(out6['loss']))
conv_perte_test_6 = np.ones(len(out6['val_loss']))*np.min(out6['val_loss'])
conv_entr_6 = np.ones(len(out6['loss']))*taux_entr_6
conv_val_6 = np.ones(len(out6['loss']))*taux_val_6

# Plot Function de perte
fig, ax = plt.subplots()
ax.plot(iterations,conv_perte_6, c = 'k', ls = '--', label = "convergence souhaitée")
ax.plot(iterations,conv_perte_test_6, c = 'r', ls = '--', label = "convergence test")
ax.plot(out6['loss'], c='b', ls='-', label="train")
ax.plot(out6['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_6.pdf'
ax.legend()
plt.savefig(plotnom)

# Plot Taux de succès
fig, ax = plt.subplots()
ax.plot(iterations,conv_entr_6, c = 'b', ls = '--', label = "train convergence")
ax.plot(iterations,conv_val_6, c = 'r', ls = '--', label = "test convergence")
ax.plot(iterations, out6['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out6['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_6.pdf'
ax.legend()
plt.savefig(plotnom)

taux_entr_6 = np.max(out6['accuracy'])
taux_val_6 = np.max(out6['val_accuracy'])
taux_max_6 = np.max([out6['accuracy'], out6['val_accuracy']])

# Taux de succès
print("Pour le modèle 6, avec", len(out6['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_6)
print("La dernière valeur est :", out6['val_accuracy'][-1])
