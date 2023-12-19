import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
import os
import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

mnist=tf.keras.datasets.mnist

(x_temp,y_temp),(x_test,y_test)=mnist.load_data()

x_temp=tf.keras.utils.normalize(x_temp,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

x_train, x_cv, y_train, y_cv=train_test_split(x_temp, y_temp, test_size=0.2, random_state=100)

model= tf.keras.models.Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(128, activation='relu'))

model.add(Dense(20, activation='relu'))

model.add(Dense(10, activation='linear'))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=8)

loss_cv, accuracy_cv=model.evaluate(x_cv,y_cv)
loss_test, accuracy_test=model.evaluate(x_test, y_test)


print(f"Cross Validation Loss: {loss_cv}, Test Loss: {loss_test}")
print(f"Cross Validation Accuracy: {accuracy_cv}, Test Accuracy: {accuracy_test}")


with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)

