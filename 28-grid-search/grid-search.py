
import pandas as pd
import numpy as np

# THE PURPOSE IS TO VALIDATE ACCURACY FROM MODEL PREDICTIONS

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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Fit the Training Model on Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
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
print('Test Accuracy')
print(score)
print('')
print('Confusion Matrix')
print(cm)

# Apply K-Fold Cross Validation
# "K" number of folds "cv" is HARD CODED
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print('')
print(f'K-Fold Accuracy {accuracies.mean()*100:.2f} %')
print('')
print(f'Standard Deviation {accuracies.std()*100:.2f} %')

# Apply Grid Search to find the best model and parameters
# Argument size varies according to values given like "kernel"
# "K" number of folds "cv" is HARD CODED
# "n_jobs" is HARD CODED for local machine
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv = 10,
                           n_jobs=-1)
grid_search.fit(x_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
print('')
print(f'Best Grid Accuracy {best_score*100:.2f} %')
print('')
print(f'Best Grid Parameters {best_parameters}')

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

# plt.title('Kernel SVM (Training Model)')
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

# plt.title('Kernel SVM (Test Model)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
