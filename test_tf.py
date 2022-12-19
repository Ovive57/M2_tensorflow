import tensor_flow as t_f

import numpy as np
import matplotlib.pyplot as plt

import style

path2plot = "plots/"
plt.style.use(style.style1)

##### Partie tensor flow :
# Pour cette partie j'importe le fichier out avec les data dedans pour faire l'analyse
# Pour éviter de refaire les calculs longs de tensor flow
# Pour ça je commente le import tensor_flow en haut sinon ca démarre tout seul 



# Ouvre le fichier avec le dictionaire out dedans:

#with open('out.pkl', 'rb') as f:
 #   out = pickle.load(f)
"""
print("\n\n\n Ca commence là:")

out=np.load('out.npy',allow_pickle='TRUE').item()
"""

#out.history # ne fait rien

#print("Clés associées au dictionnaire .history:", out.history.keys())

#print("Clés associées au dictionnaire out.history:", t_f.out.keys())
print("Clés associées au dictionnaire out.history:", t_f.out.history.keys())
# loss, accuracy, val_loss, val_accuracy

# A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable)

# Loss = loss values doit tendre à 0 (Loss = perte) Fonction de pertre
# Acc = metrics values doit tendre à 1 (Accuracy = justesse) Taux de succès

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/


print("Loss ", len(t_f.out.history['loss']))
print("Acc ", len(t_f.out.history['accuracy']))
print("Val Loss", len(t_f.out.history['val_loss']))
print("Val acc ", len(t_f.out.history['val_accuracy']))
print("Taille des listes des 4 entrées du dict: 300 (= au nombre d'époques)")

"""
print("Loss ", len(out['loss']))
print("Acc ", len(out['accuracy']))
print("Val Loss", len(out['val_loss']))
print("Val acc ", len(out['val_accuracy']))
print("Taille des listes des 4 entrées du dict: 300 (= au nombre d'époques)")
"""
iterations = np.arange(len(t_f.out.history['loss']))

#iterations = np.arange(len(out['loss']))


# Evolution de la fonction de perte en fonction des itérations:


fig = plt.figure(figsize=(25, 25))

# Pour l'échantillon d'entrainement:

plt.subplot(2, 2, 1)
plt.plot(iterations, t_f.out.history['loss'])
#plt.plot(iterations, out['loss'])
plt.title("Evolution de la fonction de perte pour l'échantillon d'entrainement")
plt.xlabel("Itérations")
plt.ylabel("Fonction de perte")

# Pour l'échantillon de validation:

plt.subplot(2, 2, 2)
plt.plot(iterations, t_f.out.history['accuracy'])
#plt.plot(iterations, out['accuracy'])
plt.title("Evolution du taux de succès pour l'échantillon d'entrainement")
plt.xlabel("Itérations")
plt.ylabel("Taux de succès")

# Evolution du taux de succès en fonction des itérations:
# Pour l'échantillon d'entrainement:

plt.subplot(2, 2, 3)
plt.plot(iterations, t_f.out.history['val_loss'])
#plt.plot(iterations, out['val_loss'])
plt.title("Evolution de la fonction de perte pour l'échantillon de validation")
plt.xlabel("Itérations")
plt.ylabel("Fonction de perte")

# Pour l'échantillon de validation:

plt.subplot(2, 2, 4)
plt.plot(iterations, t_f.out.history['val_accuracy'])
#plt.plot(iterations, out['val_accuracy'])
plt.title("Evolution du taux de succès pour l'échantillon de validation")
plt.xlabel("Itérations")
plt.ylabel("Taux de succès")

#plt.show()
plotnom = path2plot + 'Evolutions.pdf'
plt.savefig(plotnom)

#https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# Faire plots train/test

# val = test
# no val = train
"""
plt.plot(out.history['accuracy'])
plt.plot(out.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(out.history['loss'])
plt.plot(out.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""
# Le taux de succès maximale atteint:


#taux_max = np.max([out.history['accuracy'], out.history['val_accuracy']])
taux_entr = np.max(t_f.out.history['accuracy'])
taux_val = np.max(t_f.out.history['val_accuracy'])
taux_max = np.max([t_f.out.history['accuracy'], t_f.out.history['val_accuracy']])

print("Le taux de succès maximal atteint est:", taux_max, " et il doit tendre vers 1")
print("Le taux de succès maximal d'entrainement atteint est:", taux_entr, " et il doit tendre vers 1")
print("Le taux de succès maximal de validation atteint est: AVEC 300 EPOCH", taux_val, " et il doit tendre vers 1")


# A-t-il convergé ? Presque ! Mais pas à 10⁻3 près. En fait je pense que la phrase veut dire qu'il faut s'approcher à 1 avec une difference de maximum 10⁻3, c'est à dire, avoir un taux de succès de 0.99... minimum.

# Je pense que c'est celle de validation qui doit aller vers 1.
taux_entr_c = np.max(t_f.accuracy_c)
taux_val_c = np.max(t_f.val_accuracy_c)
taux_max_c = np.max([t_f.accuracy_c, t_f.val_accuracy_c])

print("Le taux de succès maximal atteint est:", taux_max_c, " et il doit tendre vers 1")
print("Le taux de succès maximal d'entrainement atteint est:", taux_entr_c, " et il doit tendre vers 1")
print("Le taux de succès maximal de validation atteint est: AVEC 1000 EPOCH", taux_val_c, " et il doit tendre vers 1")

# Il arrive jamais à 0.99, j'ai essayé avec 10000 et ça ameliore les 300 de 0.9412 a 0.9427, vraiment pas beaucoup. En plus ça se voit que se surentraîne, le taux de succès d'entraînement augmente bien, vers 0.9725, mais la dernière valeur pour le taux de succès de validation a descendu jusqu'à 0.9252 dans les derniers epoch. loss arrive a 0.0973 et la validation est a 0.4753, meme si je l'ai vu descendre à 0.9 aussi. Je pense qu'on peut envoyer un mail à Florian sur ça.




