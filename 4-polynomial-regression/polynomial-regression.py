
import pandas as pd
import numpy as np

# PURPOSE IS TO PREDICT SALARY

# Simple Linear vs. Polynomial
# 1. "Simple" is visualized as a straight line.
# 2. "Polynomial" is visualized as an exponential curve.

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

# Fit the Linear Regression to the entire dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fit the Polynomial Regression to the entire dataset
from sklearn.preprocessing import PolynomialFeatures

# "degree=4" is quite high and can lead to overfitting
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

# Fit the Polynomial values of X to new Linear Regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualize the Linear Regression
# as PROOF this data-relationship is not linear
import matplotlib.pyplot as plt
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Inaccurate Salary Prediction (Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualize the Polynomial Regression
import matplotlib.pyplot as plt
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Accurate Salary Prediction (Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Higher Resolution visuzliation of Polynomial Regression
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('High Resolution Salary Prediction (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Test inaccuracy of new values using Linear Regression
# Should be around $160k
print('Inaccurate: Should be around 160k')
print(lin_reg.predict([[6.5]]))

# Test accuracy of new values using Polynomial Regression
# Should be around $160k
print('Accurate: This is around 160k')
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))