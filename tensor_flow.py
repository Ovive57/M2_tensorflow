import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

pathfiles = "files/"
num_classes = 10

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Normalisation :

X_train = X_train.astype("float32")/np.max(X_test) # np.max(X_test) = np.max(X_train) = 255
X_test = X_test.astype("float32")/np.max(X_test)


# Aplatissement des images :

X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))

# Transformation des classes en matrices binaires de taille 10:

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


#### MODELE 1 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones. 
# Taux d'apprentissage de 0.01.
# 300 époques.
# batch_size 60000.

input_shape = X_train.shape[1] #784

model1 = keras.Sequential()
model1.add(keras.Input(shape=input_shape))
model1.add(layers.Dense(num_classes, activation="relu", name = "cachee"))
model1.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

print("\nMODELE 1 : lambda = 0.01, 300 époques.\n")
model1.summary() # Affichage modèle 1

# Créer un objet opt pour la minimisation de la fonction de perte:

opt = keras.optimizers.Adam(learning_rate=0.01)

# Associe l'objet opt à model avec la méthode compile:

model1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Entraînement du réseau de neuronnes avec la méthode fit:

out1 = model1.fit(X_train, Y_train, batch_size=len(X_train), epochs=300, validation_data=(X_test, Y_test))

np.save(pathfiles + "out1.npy", out1.history)


#### MODELE 2 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 1000 époques, pour essayer qu'il converge à 10⁻3 près.
# batch_size 60000.

model2 = keras.Sequential()
model2.add(keras.Input(shape=input_shape))
model2.add(layers.Dense(num_classes, activation="relu", name = "cachee"))
model2.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

print("\nMODELE 2 : lambda = 0.01, 1000 époques.\n")
model2.summary()

opt2 = keras.optimizers.Adam(learning_rate=0.01)
model2.compile(loss='categorical_crossentropy', optimizer=opt2, metrics=['accuracy'])
out2 = model2.fit(X_train, Y_train, batch_size=len(X_train), epochs=1000, validation_data=(X_test, Y_test))

np.save(pathfiles + "out2.npy", out2.history)


#### MODELE 2 bis ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 10000 époques, pour essayer qu'il converge à 10⁻3 près où voir le surentraînement.
# batch_size 60000.

model2b = keras.Sequential()
model2b.add(keras.Input(shape=input_shape))
model2b.add(layers.Dense(num_classes, activation="relu", name = "cachee"))
model2b.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

print("\nMODELE 2 bis: lambda = 0.01, 10000 époques.\n")
model2b.summary()

opt2b = keras.optimizers.Adam(learning_rate=0.01)
model2b.compile(loss='categorical_crossentropy', optimizer=opt2b, metrics=['accuracy'])
out2b = model2b.fit(X_train, Y_train, batch_size=len(X_train), epochs=10000, validation_data=(X_test, Y_test))

np.save(pathfiles + "out2b.npy", out2b.history)


#### MODELE 3 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.2.
# 300 époques.
# batch_size 60000.

model3 = keras.Sequential()
model3.add(keras.Input(shape=input_shape))
model3.add(layers.Dense(num_classes, activation="relu", name = "cachee"))
model3.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

print("\nMODELE 3 : lambda = 0.2, 300 époques.\n")
model3.summary()

opt3 = keras.optimizers.Adam(learning_rate=0.2)
model3.compile(loss='categorical_crossentropy', optimizer=opt3, metrics=['accuracy'])
out3 = model3.fit(X_train, Y_train, batch_size=len(X_train), epochs=300, validation_data=(X_test, Y_test))

np.save(pathfiles + "out3.npy", out3.history)


#### MODELE 4 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10*50 = 500 neurones, une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 300 époques.
# batch_size 60000.

model4 = keras.Sequential()
model4.add(keras.Input(shape=input_shape))
model4.add(layers.Dense(num_classes*50, activation="relu", name = "cachee"))
model4.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

print("\nMODELE 4 : lambda = 0.01, 300 époques.\n")
model4.summary()

opt4 = keras.optimizers.Adam(learning_rate=0.01)
model4.compile(loss='categorical_crossentropy', optimizer=opt4, metrics=['accuracy'])
out4 = model4.fit(X_train, Y_train, batch_size=len(X_train), epochs=300, validation_data=(X_test, Y_test))

np.save(pathfiles + "out4.npy", out4.history)


#### MODELE 5 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10*50 = 500 neurones, une deuxième couche cachée de 700 neurones,
# une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 200 époques.
# batch_size 60000.

model5 = keras.Sequential()
model5.add(keras.Input(shape=input_shape))
model5.add(layers.Dense(num_classes*50, activation="relu", name = "cachee1"))
model5.add(layers.Dense(700, activation="relu", name = "cachee2"))
model5.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

print("\nMODELE 5 : lambda = 0.01, 200 époques.\n")
model5.summary()

opt5 = keras.optimizers.Adam(learning_rate=0.01)
model5.compile(loss='categorical_crossentropy', optimizer=opt5, metrics=['accuracy'])
out5 = model5.fit(X_train, Y_train, batch_size=len(X_train), epochs=200, validation_data=(X_test, Y_test))

np.save(pathfiles + "out5.npy", out5.history)


#### MODELE 6 ####

# Modèle avec une couche d'entrée de 784 neurones, une couche cachée de 10*50 = 500 neurones, une deuxième couche cachée de 700 neurones,
# une couche de sortie de 10 neurones.
# Taux d'apprentissage de 0.01.
# 200 époques.
# batch_size 60000/10 = 6000.

model6 = keras.Sequential()
model6.add(keras.Input(shape=input_shape))
model6.add(layers.Dense(num_classes*50, activation="relu", name = "cachee1"))
model6.add(layers.Dense(700, activation="relu", name = "cachee2"))
model6.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

print("\nMODELE 6 : lambda = 0.01, 200 époques.\n")
model6.summary()

opt6 = keras.optimizers.Adam(learning_rate=0.01)
batch_size = int(len(X_train)/10)
model6.compile(loss='categorical_crossentropy', optimizer=opt6, metrics=['accuracy'])
out6 = model6.fit(X_train, Y_train, batch_size=batch_size, epochs=200, validation_data=(X_test, Y_test))

np.save(pathfiles + "out6.npy", out6.history)