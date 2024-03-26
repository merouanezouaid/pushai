import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load the data
X_correct = np.load("dataset/labels/correct.npy")
y_correct = np.ones(len(X_correct))  # labels for correct push-ups

X_incorrect = np.load("dataset/labels/incorrect.npy")
y_incorrect = np.zeros(len(X_incorrect))  # labels for incorrect push-ups

# Combine the data
X = np.concatenate((X_correct, X_incorrect))
y = np.concatenate((y_correct, y_incorrect))

# Preprocess the data
X = X.reshape(X.shape[0], 100, -1)  # reshape the data

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(100, 99))) 
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax')) 


 # Assuming binary classification

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Evaluate the model
val_loss, val_acc = model.evaluate(X_test, y_test)
print("Loss:", val_loss)
print("Accuracy:", val_acc)

# Save the model
model.save("model.h5")