
import pandas as pd

# PURPOSE IS TO PREDICT SALARY

# This example dataset assumes that "Job Title"
# is the leading cause of high or low Salary
# and has not gone through an optimized variable selection process

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Below code assumes Depedent Variable is in the last column

# Independent variable "Job ID" values as array
# Target everything except the last column AND the first column
x = dataset.iloc[:, 1:-1].values
# print(x)

# Dependent variable "Salary" values as array
y = dataset.iloc[:, -1].values
# print(y)

y = y.reshape(len(y), 1)
# print(y)

# Feature Scaling to prevent units of measure from mapping incorrectly
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
# print(x)
# print(y)

# WARNING: This example trains the entire dataset
#          In reality we split the dataset into Train/Test

# Fit the entire dataset on SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

# Predict one new result
# Feature Scaling "sc_x" since dataset was scaled
# Reverse Scaling "inverse_transform" to print prediction
prediction = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

# Visualize the Support Vector Regression
import matplotlib.pyplot as plt
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color='blue')
plt.title('Salary Prediction (SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Higher Resolution visuzliation of Support Vector Regression
import numpy as np
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color='blue')
plt.title('High Resolution Salary Prediction (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
