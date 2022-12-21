import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt

num_classes = 10


(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data() 

# np.max(X_test) == 255

# Normalisation :

X_train = X_train.astype("float32")/np.max(X_test)
X_test = X_test.astype("float32")/np.max(X_test)


# Aplatissement des images :

X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))




# Transformation des classes en matrices binaires de taille 10:
 
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


input_shape = X_train.shape[1] #784

model = keras.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Dense(num_classes, activation="relu", name = "cachee"))
model.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

model.summary()


# Créer un objet opt pour la minimisation de la fonction de perte:

opt = keras.optimizers.Adam(learning_rate=0.01)



# Associe l'objet opt à model :(même page https://keras.io/api/optimizers/ )
# + ajout de 'metrics' comme demandé dans l'énoncé

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# Entraine le réseau de neuronnes

out = model.fit(X_train, Y_train, batch_size=len(X_train), epochs=300, validation_data=(X_test, Y_test))


np.save("out.npy", out.history)


loss = out.history['loss']
accuracy = out.history['accuracy']
val_loss = out.history['val_loss']
val_accuracy = out.history['val_accuracy']



# Model avec plus d'epoques pour essayer qu'il converge à 10⁻3 près, ça marche moyen (c = qui converge)
# Le modèle doit être refait à chaque fois pour ne pas reprendre le taux de succès précédent

model_c = keras.Sequential()
model_c.add(keras.Input(shape=input_shape))
model_c.add(layers.Dense(num_classes, activation="relu", name = "cachee"))
model_c.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

model_c.summary()

opt_c = keras.optimizers.Adam(learning_rate=0.01)

model_c.compile(loss='categorical_crossentropy', optimizer=opt_c, metrics=['accuracy'])


out_c = model_c.fit(X_train, Y_train, batch_size=len(X_train), epochs=1000, validation_data=(X_test, Y_test))

np.save("out_c.npy", out_c.history)

loss_c = out_c.history['loss']
accuracy_c = out_c.history['accuracy']
val_loss_c = out_c.history['val_loss']
val_accuracy_c = out_c.history['val_accuracy']



# Même entraînement avec un taux d'apprentisage de 0.2 (02 = 0.2 taux)

model_02 = keras.Sequential()
model_02.add(keras.Input(shape=input_shape))
model_02.add(layers.Dense(num_classes, activation="relu", name = "cachee"))
model_02.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

model_02.summary()

opt_02 = keras.optimizers.Adam(learning_rate=0.2)

model_02.compile(loss='categorical_crossentropy', optimizer=opt_02, metrics=['accuracy'])
out_02 = model_02.fit(X_train, Y_train, batch_size=len(X_train), epochs=300, validation_data=(X_test, Y_test))

np.save("out_02.npy", out_02.history)

## Nouveau réseau de néurones avec 10*50 = 500 neurones dans la couche cachée (500 = 500 neurones):

model_500 = keras.Sequential()
model_500.add(keras.Input(shape=input_shape))
model_500.add(layers.Dense(num_classes*50, activation="relu", name = "cachee"))
model_500.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

model_500.summary()

opt_500 = keras.optimizers.Adam(learning_rate=0.01)

model_500.compile(loss='categorical_crossentropy', optimizer=opt_500, metrics=['accuracy'])
out_500 = model_500.fit(X_train, Y_train, batch_size=len(X_train), epochs=300, validation_data=(X_test, Y_test))

np.save("out_500.npy", out_500.history)

## Nouveau réseau avec une nouvelle couche cachée de 700 neurones et 200 epoques (nc = nouvelle couche)

model_nc = keras.Sequential()
model_nc.add(keras.Input(shape=input_shape))
model_nc.add(layers.Dense(num_classes*50, activation="relu", name = "cachee1"))
model_nc.add(layers.Dense(700, activation="relu", name = "cachee2"))
model_nc.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

model_nc.summary()

opt_nc = keras.optimizers.Adam(learning_rate=0.01)

model_nc.compile(loss='categorical_crossentropy', optimizer=opt_nc, metrics=['accuracy'])
out_nc = model_nc.fit(X_train, Y_train, batch_size=len(X_train), epochs=200, validation_data=(X_test, Y_test))

np.save("out_nc.npy", out_nc.history)


# Modification variable batch_size en divisant par 10, pour 200 epoques. (bs = batch_size)

model_bs = keras.Sequential()
model_bs.add(keras.Input(shape=input_shape))
model_bs.add(layers.Dense(num_classes*50, activation="relu", name = "cachee1"))
model_bs.add(layers.Dense(700, activation="relu", name = "cachee2"))
model_bs.add(layers.Dense(num_classes, activation="softmax", name = "sortie"))

model_bs.summary()

opt_bs = keras.optimizers.Adam(learning_rate=0.01)
batch_size = int(len(X_train)/10)

model_bs.compile(loss='categorical_crossentropy', optimizer=opt_bs, metrics=['accuracy'])
out_bs = model_bs.fit(X_train, Y_train, batch_size=batch_size, epochs=200, validation_data=(X_test, Y_test))

np.save("out_bs.npy", out_bs.history)


























