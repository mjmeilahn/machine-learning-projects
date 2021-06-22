
import pandas as pd
import numpy as np

# THE PURPOSE IS TO PREDICT PURCHASE RATE

# This example dataset assumes that "Age, Estimated Salary"
# are the leading causes of Purchase Rate
# and has not gone through an optimized variable selection process

# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Below code assumes Depedent Variable is in the last column

# Independent variables "Age, Estimated Salary" values as array
x = dataset.iloc[:, :-1].values
# print(x)

# Dependent variable "Purchased" values as array
y = dataset.iloc[:, -1].values
# print(y)

# Split data into Training Model & Test Model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Always apply Feature Scaling AFTER the data split
# To avoid overfitting the Training Model

# Feature Scaling is not applicable to all Machine Learning
# Such as categorical variables

# OPTIONAL: Feature Scaling is not required for Logistic Regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Fit the Training Model on K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

# Predict one new result
# print(classifier.predict(scaler.transform([[30, 87000]])))

# Predict Test Model results
y_pred = classifier.predict(x_test)
# print('Predicted vs. Actual')
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Print Accuracy & Confusion Matrix results
from sklearn.metrics import confusion_matrix, accuracy_score
score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print('Accuracy Score')
print(score)
print('')
print('Confusion Matrix')
print(cm)


# WARNING: Below code eats a lot of CPU at runtime

# Visualize Training Model results
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# x_set, y_set = scaler.inverse_transform(x_train), y_train
# x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 10, stop = x_set[:,0].max() + 10, step = 0.25), np.arange(start = x_set[:,1].min() - 1000, stop = x_set[:,1].max() + 1000, step = 0.25))

# plt.contour(x1, x2, classifier.predict(scaler.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# plt.xlim(x1.min(), x1.max())
# plt.ylim(x2.min(), x2.max())

# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

# plt.title('Logistic Regression (Training Model)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()

# Visualize Test Model results
# x_set, y_set = scaler.inverse_transform(x_test), y_test
# x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 10, stop = x_set[:,0].max() + 10, step = 0.25), np.arange(start = x_set[:,1].min() - 1000, stop = x_set[:,1].max() + 1000, step = 0.25))

# plt.contour(x1, x2, classifier.predict(scaler.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# plt.xlim(x1.min(), x1.max())
# plt.ylim(x2.min(), x2.max())

# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

# plt.title('Logistic Regression (Test Model)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
