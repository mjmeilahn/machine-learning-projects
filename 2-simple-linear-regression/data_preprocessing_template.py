
import pandas as pd

# PURPOSE IS TO PREDICT SALARY

# This example dataset assumes that "Years of Experience"
# is the leading cause of Salary and has not gone
# through an optimized process regarding P-values or Adj. R-Squared

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Below code assumes Depedent Variable is in the last column

# Independent Variable "Years of Experience" values as matrix
x = dataset.iloc[:, :-1].values
# print(x)

# Dependent Variable "Salary" values as matrix
y = dataset.iloc[:, -1].values
# print(y)

# Split data into Training Model & Test Model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)