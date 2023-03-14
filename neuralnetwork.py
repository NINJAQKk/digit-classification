from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import six

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
print(x_test[0])

classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

input_shape = (28, 28, 1)

x_train = x_train / 255.0

x_test = x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train, classes.shape[0])
y_test = tf.keras.utils.to_categorical(y_test, classes.shape[0])

model = tf.keras.Sequential([
  tf.keras.Input(shape=input_shape),
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(.5),
  tf.keras.layers.Dense(classes.shape[0], activation='softmax')
])

print(model.summary())

batch_size = 100
epochs = 5

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=2, shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
