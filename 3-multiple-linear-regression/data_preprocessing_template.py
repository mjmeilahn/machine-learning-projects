
import pandas as pd

# PURPOSE IS TO PREDICT PROFIT

# This example dataset assumes that "R&D Spend, Administration...
# Marketing Spend and State" are the leading cause of Profit
# and has not gone through an optimized variable selection process

# Import the dataset
dataset = pd.read_csv('50_Startups.csv')

# Below code assumes Depedent Variable is in the last column

# Independent Variables "R&D Spend, Administration...
# Marketing Spend, State" values as array
x = dataset.iloc[:, :-1].values
# print(x)

# Dependent Variable "Profit" values as array
y = dataset.iloc[:, 4].values
# print(y)