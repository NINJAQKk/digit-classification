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

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images = train_images / 255.0

# test_images = test_images / 255.0

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
#     keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
#     keras.layers.Dense(10, activation='softmax') # output layer (3)
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

# print('Test accuracy:', test_acc)

# predictions = model.predict(test_images)