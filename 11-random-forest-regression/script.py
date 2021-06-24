
import pandas as pd
import numpy as np

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

# WARNING: This example trains the entire dataset
#          In reality we split the dataset into Train/Test

# Fit the dataset on the Random Forest
from sklearn.ensemble import RandomForestRegressor

# Take random samples from 10 Decision Trees
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

# Predict one new result
regressor.predict([[6.5]])

# Visualize the Random Forest Regression
import matplotlib.pyplot as plt
# plt.scatter(x, y, color='red')
# plt.plot(x, regressor.predict(x), color='blue')
# plt.title('Salary Prediction (Random Forest Regression')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()

# Higher Resolution visuzliation of Random Forest Regression
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('High Resolution Salary Prediction (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()