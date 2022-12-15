import numpy as np
import tensorflow as tf
from tensorflow import keras



num_classes = 10

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()  # meme chose

# np.max(X_test) == 255

X_train = X_train.astype("float32")/np.max(X_test)
X_test = X_test.astype("float32")/np.max(X_test)


X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))

print(X_train)
Y_train = keras.utils.to_categorical(Y_train, num_classes)

Y_test = keras.utils.to_categorical(Y_test, num_classes)


# https://keras.io/examples/vision/mnist_convnet/ pour la doc
