import numpy as np
import matplotlib.pyplot as plt

import style

path2plot = "plots/"
pathfiles = "files/"
plt.style.use(style.style1)

##### Partie tensor flow :


# Ouvre le fichier contenant le dictionaire out :

out1 = np.load(pathfiles + 'out1.npy', allow_pickle='TRUE').item()


print("Clés associées au dictionnaire out.history:", out1.keys())
# loss, accuracy, val_loss, val_accuracy



print("Loss ", len(out1['loss']))
print("Acc ", len(out1['accuracy']))
print("Val Loss", len(out1['val_loss']))
print("Val acc ", len(out1['val_accuracy']))
print("Taille des listes des 4 entrées du dict: 300 (= au nombre d'époques)")


####### MODELE 1 #########

print("\n Modèle Initial avec 300 itérations et un taux d'apprentissage 0.01")

iterations = np.arange(len(out1['loss']))
conv_perte = np.zeros(len(out1['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out1['loss'], c='b', ls='-', label="train")
ax.plot(out1['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_1.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()


conv = np.ones(len(out1['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(out1['accuracy'], c='b', ls='-', label="train")
ax.plot(out1['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_1.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()


# Le taux de succès maximale atteint:
# Pour moi le taux de succès qui nous interese est celui de la validation. 
# Le taux des train va s'ameliorer tout le temps si on entraine de plus en plus, i.e. il va de mieux en mieux trouver les images de l'entrainement, 
# mais à nous ça nous intèrese qu'un fois il est entraîné, il trouve le mieux possible les images de test, 
# ce sont les images de la validation. Si on ameliore trop le taux d'entraînement on risque de faire un surentraînement. 
# Dit moi si je me suis bien expliqué là.

# Constance: Je suis d'accord que c'est le taux de validation
# Il faudrait mettre la valeur max pour le 10 000 + le graphe

taux_entr = np.max(out1['accuracy'])
taux_val = np.max(out1['val_accuracy'])
taux_max = np.max([out1['accuracy'], out1['val_accuracy']])

#print("Le taux de succès maximal atteint est:", taux_max, " et il doit tendre vers 1")
#print("Le taux de succès maximal de validation atteint est:", taux_val, " et il doit tendre vers 1")
#print("Le taux de succès maximal d'entrainement atteint est:", taux_entr, " et il doit tendre vers 1")

print("Avec", len(out1['accuracy']), " epochs, le taux de succès maximal est de :", taux_val)
print("La dernière valeur est :", out1['val_accuracy'][-1])

# A-t-il convergé ? Presque ! Mais pas à 10⁻3 près.
# En fait je pense que la phrase veut dire qu'il faut s'approcher à 1 avec une difference de maximum 10⁻3, c'est à dire, avoir un taux de succès de 0.99... minimum.

# Je pense que c'est celle de validation qui doit aller vers 1.


####### MODELE 2 #########

print("\n Modèle 2 avec 1000 itérations et un taux d'apprentissage 0.01")

out2=np.load(pathfiles + 'out2.npy',allow_pickle='TRUE').item()

iterations = np.arange(len(out2['loss']))
conv_perte = np.zeros(len(out2['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out2['loss'], c='b', ls='-', label="train")
ax.plot(out2['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_2.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out2['val_accuracy']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out2['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out2['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_2.pdf'
ax.legend()
plt.savefig(plotnom)
plt.show()




taux_entr_2 = np.max(out2['accuracy'])
taux_val_2 = np.max(out2['val_accuracy'])
taux_max_2 = np.max([out2['accuracy'], out2['val_accuracy']])

print("Le taux de succès maximal atteint est:", taux_max_2, " et il doit tendre vers 1")
print("Le taux de succès maximal de validation atteint est:", taux_val_2, " et il doit tendre vers 1")
print("Le taux de succès maximal d'entrainement atteint est:", taux_entr_2, " et il doit tendre vers 1")

print("Avec", len(out2['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_2)

# Il arrive jamais à 0.99, j'ai essayé avec 10000 et ça ameliore les 300 de 0.9412 a 0.9427, vraiment pas beaucoup. 
# En plus ça se voit que se surentraîne, le taux de succès d'entraînement augmente bien, vers 0.9725, 
# mais la dernière valeur pour le taux de succès de validation a descendu jusqu'à 0.9252 dans les derniers epoch. 
#loss arrive a 0.0973 et la validation est a 0.4753, meme si je l'ai vu descendre à 0.9 aussi. 
#Je pense qu'on peut envoyer un mail à Florian sur ça.



####### MODELE 2 bis #########

print("\n Modèle 2 avec 10 000 itérations et un taux d'apprentissage 0.01")


out2b=np.load(pathfiles + 'out2b.npy',allow_pickle='TRUE').item()

iterations = np.arange(len(out2b['loss']))
conv_perte = np.zeros(len(out2b['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out2b['loss'], c='b', ls='-', label="train")
ax.plot(out2b['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_2b.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out2b['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out2b['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out2b['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_2b.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()


taux_entr_2b = np.max(out2b['accuracy'])
taux_val_2b = np.max(out2b['val_accuracy'])
taux_max_2b = np.max([out2b['accuracy'], out2b['val_accuracy']])

print("Le taux de succès maximal atteint est:", taux_max_2b, " et il doit tendre vers 1")
print("Le taux de succès maximal de validation atteint est:", taux_val_2b, " et il doit tendre vers 1")
print("Le taux de succès maximal d'entrainement atteint est:", taux_entr_2b, " et il doit tendre vers 1")

print("Avec", len(out2b['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_2b)



######### MODELE 3 ##########

print("\n Modèle 3 avec 300 itérations et un taux d'apprentissage 0.2")

# # Même entraînement avec un taux d'apprentisage de 0.2:

out3=np.load(pathfiles + 'out3.npy', allow_pickle='TRUE').item()


iterations = np.arange(len(out3['loss']))
conv_perte = np.zeros(len(out3['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out3['loss'], c='b', ls='-', label="train")
ax.plot(out3['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_3.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out3['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out3['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out3['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_3.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()




taux_entr_3 = np.max(out3['accuracy'])
taux_val_3 = np.max(out3['val_accuracy'])
taux_max_3 = np.max([out3['accuracy'], out3['val_accuracy']])

print("Avec", len(out3['accuracy']), " epochs, et le taux d'apprentissage = 0.2, le taux de succès maximal est de :", taux_val_3) # Il est pire



######### MODELE 4 ##########

# Nouveau réseau de neurones avec 10*50 = 500 neurones dans la couche cachée (500 = 500 neurones):

out4=np.load(pathfiles + 'out4.npy', allow_pickle='TRUE').item()


iterations = np.arange(len(out4['loss']))
conv_perte = np.zeros(len(out4['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out4['loss'], c='b', ls='-', label="train")
ax.plot(out4['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_4.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out4['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out4['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out4['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_4.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

taux_entr_4 = np.max(out4['accuracy'])
taux_val_4 = np.max(out4['val_accuracy'])
taux_max_4 = np.max([out4['accuracy'], out4['val_accuracy']])

print("Avec", len(out4['accuracy']), " epochs, avec 500 neurones dans la couche cachée, le taux de succès maximal est de :", taux_val_4)



######### MODELE 5 ##########

## Nouveau réseau avec une nouvelle couche cachée de 700 neurones et 200 epoques (nc = nouvelle couche)

out5=np.load(pathfiles + 'out5.npy',allow_pickle='TRUE').item()


iterations = np.arange(len(out5['loss']))
conv_perte = np.zeros(len(out5['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out5['loss'], c='b', ls='-', label="train")
ax.plot(out5['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_5.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out5['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out5['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out5['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du tauc de succès")
plotnom = path2plot + 'taux_succes_5.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

taux_entr_5 = np.max(out5['accuracy'])
taux_val_5 = np.max(out5['val_accuracy'])
taux_max_5 = np.max([out5['accuracy'], out5['val_accuracy']])

print("Le taux de succès maximal de validation atteint est:", taux_val_5, " et il doit tendre vers 1")
print("Le taux de succès maximal d'entrainement atteint est:", taux_entr_5, " et il doit tendre vers 1")

print("Avec", len(out5['accuracy']), " epochs, et une deuxième couche cahcée, le taux de succès maximal est de :", taux_val_5)


######### MODELE 6 ##########

## Modification variable batch_size en divisant par 10, pour 200 epoques. (bs = batch_size)

out6=np.load(pathfiles + 'out6.npy',allow_pickle='TRUE').item()


iterations = np.arange(len(out6['loss']))
conv_perte = np.zeros(len(out6['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out6['loss'], c='b', ls='-', label="train")
ax.plot(out6['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_6.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out6['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out6['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out6['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_6.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

taux_entr_6 = np.max(out6['accuracy'])
taux_val_6 = np.max(out6['val_accuracy'])
taux_max_6 = np.max([out6['accuracy'], out6['val_accuracy']])

print("Le taux de succès maximal de validation atteint est:", taux_val_6, " et il doit tendre vers 1")
print("Le taux de succès maximal d'entrainement atteint est:", taux_entr_6, " et il doit tendre vers 1")

print("Avec", len(out6['accuracy']), " epochs, et le batch_size/10, le taux de succès maximal est de :", taux_val_6)