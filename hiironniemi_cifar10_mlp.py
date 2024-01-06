"""
Cifar 10 mlp
DATA.ML.100
Riia Hiironniemi 150271556
This code loads 5 training data batches, plots a figure that shows the NN
is learning, prints testing and training accuracys.
"""
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

import numpy as np
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Unpickle the datasets (load them into memory)
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Load dataset
datadict1 = unpickle('data_batch_1')
datadict2 = unpickle('data_batch_2')
datadict3 = unpickle('data_batch_3')
datadict4 = unpickle('data_batch_4')
datadict5 = unpickle('data_batch_5')

datadict_test = unpickle('test_batch')

X1 = datadict1["data"]
Y1 = datadict1["labels"]

X2 = datadict2["data"]
Y2 = datadict2["labels"]

X3 = datadict3["data"]
Y3 = datadict3["labels"]

X4 = datadict4["data"]
Y4 = datadict4["labels"]

X5 = datadict5["data"]
Y5 = datadict5["labels"]

x_train = np.vstack([X1, X2, X3, X4, X5])
y_train = np.concatenate([Y1, Y2, Y3, Y4, Y5])

x_test = datadict_test["data"]
y_test = datadict_test["labels"]

labeldict = unpickle('batches.meta')
label_names = labeldict["label_names"]

# Reshape
# Reshape
x_train = x_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
y_train = to_categorical(y_train, 10)

x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
y_test = to_categorical(y_test, 10)
# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Plot training loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on training and test data
train_loss, train_accuracy = model.evaluate(x_train.reshape(-1, 3072), y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test.reshape(-1, 3072), y_test, verbose=0)

print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
