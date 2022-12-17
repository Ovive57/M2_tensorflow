import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

num_classes = 10


(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data() 

# np.max(X_test) == 255

# Normalisation :

X_train = X_train.astype("float32")/np.max(X_test)
X_test = X_test.astype("float32")/np.max(X_test)

# Aplatissement des images :

X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))




# Transformation des labels en matrices binaires de taille 10:
 
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

# https://keras.io/examples/vision/mnist_convnet/ pour la doc




""" J'ai commencé à partir de là: """

# Initialiser un objet model: 

# j'ai regardé cet exemple : https://keras.io/api/models/model/  
# par ce que sur la page de base, c'est pas comme il met dans l'énoncé avec d'abord input, puis Dense puis couche finale 
# et au finale je l'ai réécrit en version condensée, donc c'est comme tu préfères
# j'ai gardé la notation 'relu' et 'softmax' ca fait plus net

# pour le input_shape j'ai galéré, au final c'est le nombre de pixels en entrée (784) tout simplement



input_shape = X_train.shape[1] #784


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(num_classes, activation="relu"),
        layers.Dense(num_classes, activation='softmax'),
    ]
)



""" Marche aussi :

inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Dense(num_classes, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

"""

# Créer un objet opt pour la minimisation de la fonction de perte:
# J'ai pris l'exemple ici: https://keras.io/api/optimizers/

opt = keras.optimizers.Adam(learning_rate=0.01)



# Associe l'objet opt à model :(même page https://keras.io/api/optimizers/ )
# + ajout de 'metrics' comme demandé dans l'énoncé

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# Entraine le réseau de neuronnes
# Là je suis retournée sur la première page: https://keras.io/examples/vision/mnist_convnet/
# pour pour validation_data sur cette page: https://www.tensorflow.org/guide/keras/train_and_evaluate
# Pour le stockage dans la variable out on voit après qu'il demande out.history donc c'est l'équivalent de history.history sur cette même page

out = model.fit(X_train, Y_train, batch_size=len(X_train), epochs=300, validation_data=(X_test, Y_test))


# Ici je le save dans un document pour pas le faire tourner à chaque fois

with open('out.pkl', 'wb') as f:
    pickle.dump(out, f)
        




