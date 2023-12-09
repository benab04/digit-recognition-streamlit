import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
import os
import cv2
import matplotlib.pyplot as plt
import pickle

mnist=tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model= tf.keras.models.Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(128, activation='relu'))

model.add(Dense(20, activation='relu'))

model.add(Dense(10, activation='linear'))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=8)

loss, accuracy=model.evaluate(x_test, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)

