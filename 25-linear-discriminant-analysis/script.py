
import pandas as pd

# THE PURPOSE IS TO FIND CORRELATIONS WITHIN DATA

# Import the Dataset
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data into Training Model & Test Model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Always apply Feature Scaling AFTER the data split
# To avoid overfitting the Training Model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Apply LDA = "n_components" is HARD CODED to dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

# Fit the Training Model on Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Print Accuracy & Confusion Matrix results
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print('Accuracy Score')
print(score)
print('')
print('Confusion Matrix')
print(cm)

# WARNING: Below code eats a lot of CPU at runtime

# Visualize Training Model results
# import numpy as np
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
# plt.xlabel('LDA1')
# plt.ylabel('LDA2')
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
# plt.xlabel('LDA1')
# plt.ylabel('LDA2')
# plt.legend()
# plt.show()