
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Data.csv')

# Below code assumes Depedent Variable is in the last column

# Independent Variables "Country, Age, Salary" values as indexes
x = dataset.iloc[:, :-1].values

# Dependent Variable "Purchased" values as indexes
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])