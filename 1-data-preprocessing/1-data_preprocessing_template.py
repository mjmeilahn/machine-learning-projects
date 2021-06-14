
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Data.csv')

# Below code assumes Depedent Variable is in the last column

# Independent Variables "Country, Age, Salary" values as matrix
x = dataset.iloc[:, :-1].values
# print(x)

# Dependent Variable "Purchased" values as matrix
y = dataset.iloc[:, 3].values
# print(y)