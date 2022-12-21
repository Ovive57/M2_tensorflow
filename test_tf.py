import numpy as np
import matplotlib.pyplot as plt

import style

path2plot = "plots/"
pathfiles = "files/"
plt.style.use(style.style1)

##### Partie tensor flow :


# Ouvre le fichier contenant le dictionaire out :

out=np.load(pathfiles + 'out.npy',allow_pickle='TRUE').item()


print("Clés associées au dictionnaire out.history:", out.keys())
# loss, accuracy, val_loss, val_accuracy



print("Loss ", len(out['loss']))
print("Acc ", len(out['accuracy']))
print("Val Loss", len(out['val_loss']))
print("Val acc ", len(out['val_accuracy']))
print("Taille des listes des 4 entrées du dict: 300 (= au nombre d'époques)")



####### MODELE 1 #########

print("\n Modèle Initial avec 300 itérations et un taux d'apprentissage 0.01")

iterations = np.arange(len(out['loss']))
conv_perte = np.zeros(len(out['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out['loss'], c='b', ls='-', label="train")
ax.plot(out['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_300.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()


conv = np.ones(len(out['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(out['accuracy'], c='b', ls='-', label="train")
ax.plot(out['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_300.pdf'
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

taux_entr = np.max(out['accuracy'])
taux_val = np.max(out['val_accuracy'])
taux_max = np.max([out['accuracy'], out['val_accuracy']])

#print("Le taux de succès maximal atteint est:", taux_max, " et il doit tendre vers 1")
#print("Le taux de succès maximal de validation atteint est:", taux_val, " et il doit tendre vers 1")
#print("Le taux de succès maximal d'entrainement atteint est:", taux_entr, " et il doit tendre vers 1")

print("Avec", len(out['accuracy']), " epochs, le taux de succès maximal est de :", taux_val)


# A-t-il convergé ? Presque ! Mais pas à 10⁻3 près. 
# En fait je pense que la phrase veut dire qu'il faut s'approcher à 1 avec une difference de maximum 10⁻3, c'est à dire, avoir un taux de succès de 0.99... minimum.

# Je pense que c'est celle de validation qui doit aller vers 1.


####### MODELE 2 #########

print("\n Modèle 2 avec 1000 itérations et un taux d'apprentissage 0.01")

out_c=np.load(pathfiles + 'out_c.npy',allow_pickle='TRUE').item()

iterations = np.arange(len(out_c['loss']))
conv_perte = np.zeros(len(out_c['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out_c['loss'], c='b', ls='-', label="train")
ax.plot(out_c['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_c.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out_c['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out_c['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out_c['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_c.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()




taux_entr_c = np.max(out_c['accuracy'])
taux_val_c = np.max(out_c['val_accuracy'])
taux_max_c = np.max([out_c['accuracy'], out_c['val_accuracy']])

#print("Le taux de succès maximal atteint est:", taux_max_c, " et il doit tendre vers 1")
#print("Le taux de succès maximal de validation atteint est:", taux_val_c, " et il doit tendre vers 1")
#print("Le taux de succès maximal d'entrainement atteint est:", taux_entr_c, " et il doit tendre vers 1")

print("Avec", len(out_c['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_c)

# Il arrive jamais à 0.99, j'ai essayé avec 10000 et ça ameliore les 300 de 0.9412 a 0.9427, vraiment pas beaucoup. 
# En plus ça se voit que se surentraîne, le taux de succès d'entraînement augmente bien, vers 0.9725, 
# mais la dernière valeur pour le taux de succès de validation a descendu jusqu'à 0.9252 dans les derniers epoch. 
#loss arrive a 0.0973 et la validation est a 0.4753, meme si je l'ai vu descendre à 0.9 aussi. 
#Je pense qu'on peut envoyer un mail à Florian sur ça.



####### MODELE 2 bis #########

print("\n Modèle 2 avec 10000 itérations et un taux d'apprentissage 0.01")


out_10000=np.load(pathfiles + 'out_10000.npy',allow_pickle='TRUE').item()

iterations = np.arange(len(out_10000['loss']))
conv_perte = np.zeros(len(out_10000['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out_10000['loss'], c='b', ls='-', label="train")
ax.plot(out_10000['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_10000.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out_10000['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out_10000['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out_10000['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du taux de succès")
plotnom = path2plot + 'taux_succes_10000.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()




taux_entr_10000 = np.max(out_10000['accuracy'])
taux_val_10000 = np.max(out_10000['val_accuracy'])
taux_max_10000 = np.max([out_10000['accuracy'], out_10000['val_accuracy']])

#print("Le taux de succès maximal atteint est:", taux_max_10000, " et il doit tendre vers 1")
#print("Le taux de succès maximal de validation atteint est:", taux_val_10000, " et il doit tendre vers 1")
#print("Le taux de succès maximal d'entrainement atteint est:", taux_entr_10000, " et il doit tendre vers 1")

print("Avec", len(out_10000['accuracy']), " epochs, le taux de succès maximal est de :", taux_val_10000)

######### MODELE 3 ##########

print("\n Modèle 3 avec 300 itérations et un taux d'apprentissage 0.2")

# # Même entraînement avec un taux d'apprentisage de 0.2:

out_02=np.load(pathfiles + 'out_02.npy',allow_pickle='TRUE').item()


iterations = np.arange(len(out_02['loss']))
conv_perte = np.zeros(len(out_02['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out_02['loss'], c='b', ls='-', label="train")
ax.plot(out_02['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_02.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out_02['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out_02['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out_02['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du tauc de succès")
plotnom = path2plot + 'taux_succes_02.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()




taux_entr_02 = np.max(out_02['accuracy'])
taux_val_02 = np.max(out_02['val_accuracy'])
taux_max_02 = np.max([out_02['accuracy'], out_02['val_accuracy']])

print("Avec", len(out_02['accuracy']), " epochs, et le taux d'apprentissage = 0.2, le taux de succès maximal est de :", taux_val_02) # Il est pire, 0.665.


######### MODELE 4 ##########

# Nouveau réseau de néurones avec 10*50 = 500 neurones dans la couche cachée (500 = 500 neurones):

out_500=np.load(pathfiles + 'out_500.npy',allow_pickle='TRUE').item()


iterations = np.arange(len(out_500['loss']))
conv_perte = np.zeros(len(out_500['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out_500['loss'], c='b', ls='-', label="train")
ax.plot(out_500['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_500.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out_500['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out_500['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out_500['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du tauc de succès")
plotnom = path2plot + 'taux_succes_500.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

taux_entr_500 = np.max(out_500['accuracy'])
taux_val_500 = np.max(out_500['val_accuracy'])
taux_max_500 = np.max([out_500['accuracy'], out_500['val_accuracy']])

print("Avec", len(out_500['accuracy']), " epochs, avec 500 neurones dans la couche cachée, le taux de succès maximal est de :", taux_val_500)


######### MODELE 5 ##########

# # Nouveau réseau avec une nouvelle couche cachée de 700 neurones et 200 epoques (nc = nouvelle couche)

out_nc=np.load(pathfiles + 'out_nc.npy',allow_pickle='TRUE').item()


iterations = np.arange(len(out_nc['loss']))
conv_perte = np.zeros(len(out_nc['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out_nc['loss'], c='b', ls='-', label="train")
ax.plot(out_nc['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_nc.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out_nc['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out_nc['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out_nc['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du tauc de succès")
plotnom = path2plot + 'taux_succes_nc.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

taux_entr_nc = np.max(out_nc['accuracy'])
taux_val_nc = np.max(out_nc['val_accuracy'])
taux_max_nc = np.max([out_nc['accuracy'], out_nc['val_accuracy']])

print("Avec", len(out_nc['accuracy']), " epochs, et une deuxième couche cahcée, le taux de succès maximal est de :", taux_val_nc)


######### MODELE 6 ##########

## Modification variable batch_size en divisant par 10, pour 200 epoques. (bs = batch_size)

out_bs=np.load(pathfiles + 'out_bs.npy',allow_pickle='TRUE').item()


iterations = np.arange(len(out_bs['loss']))
conv_perte = np.zeros(len(out_bs['loss']))


fig, ax = plt.subplots()
ax.plot(iterations,conv_perte, c = 'k', ls = '--', label = "convergence")
ax.plot(out_bs['loss'], c='b', ls='-', label="train")
ax.plot(out_bs['val_loss'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Fonction de perte")
ax.set_title("Evolution de la fonction de perte")
plotnom = path2plot + 'fonction_perte_bs.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

conv = np.ones(len(out_bs['loss']))

fig, ax = plt.subplots()
ax.plot(iterations,conv, c = 'k', ls = '--', label = "convergence")
ax.plot(iterations, out_bs['accuracy'], c='b', ls='-', label="train")
ax.plot(iterations, out_bs['val_accuracy'], c='r', ls='-', label="test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Taux de succès")
ax.set_title("Evolution du tauc de succès")
plotnom = path2plot + 'taux_succes_bs.pdf'
ax.legend()
plt.savefig(plotnom)
#plt.show()

taux_entr_bs = np.max(out_bs['accuracy'])
taux_val_bs = np.max(out_bs['val_accuracy'])
taux_max_bs = np.max([out_bs['accuracy'], out_bs['val_accuracy']])

print("Avec", len(out_bs['accuracy']), " epochs, et le batch_size/10, le taux de succès maximal est de :", taux_val_bs)




