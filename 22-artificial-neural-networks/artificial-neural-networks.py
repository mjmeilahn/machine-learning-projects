
import pandas as pd

# THE PURPOSE IS TO PREDICT CHURN RATE

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
# print(x)
# print(y)

# Set Categorical Values e.g. "Gender" as Dummy Variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
# print(x)

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)

# Split data into Training & Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Start Artificial Neural Network
import tensorflow as tf
ann = tf.keras.models.Sequential()

# WARNING: Activation Functions are HARD CODED for Binary Outcome

# Add First Hidden Layer = 6 Neurons, Rectifier
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add Second Hidden Layer = 6 Neurons, Rectifier
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add Output Layer = 1 Neuron (Binary Outcome), Sigmoid
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Train the ANN for Batch Learning, "loss" is HARD CODED for Binary
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=32, epochs=100)

# Predict a single observation - "predict" will always take two-dim. array
# Use Feature Scaling since it was implemented for Train & Test
# Only use "transform" for single observations post training phase
# NOTE: The first three values "1, 0, 0" represent France from OneHotEncoder
prediction = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
# print(prediction)

# OPTIONAL: Use equivalency to return True/False
# Based on above 95% True, below 5% False
# print(prediction > 0.5)

# Predict Test results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))