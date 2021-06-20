
import matplotlib.pyplot as plt
import pandas as pd

# PURPOSE IS TO PREDICT SALARY

# This example dataset assumes that "Years of Experience"
# is the leading cause of Salary and has not gone
# through an optimized process regarding P-values or Adj. R-Squared

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Below code assumes Depedent Variable is in the last column

# Independent Variable "Years of Experience" values as array
x = dataset.iloc[:, :-1].values
# print(x)

# Dependent Variable "Salary" values as array
y = dataset.iloc[:, -1].values
# print(y)

# Split data into Training Model & Test Model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fit Simple Linear Regression to the Training Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the Test set results
y_pred = regressor.predict(x_test)

# Visualize the Training Model results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the Test Model results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()