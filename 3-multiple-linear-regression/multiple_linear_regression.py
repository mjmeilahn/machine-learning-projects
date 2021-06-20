

import pandas as pd
import numpy as np

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
y = dataset.iloc[:, -1].values
# print(y)

# Make Dummy variables from "State" i.e. "3" in "x" dataset
# ColumnTransformer takes care of Dummy Variable Trap in categorical variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)

# Split data into Training Model & Test Model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fit Multiple Linear Regression to the Training Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the Test Model results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2) # more accurate for decimals
predictions = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1)
print('Predicted vs. Actual')
print(predictions)

# Assign predicted & actual values into lists
predicted = []
actual = []

for i in predictions:
    predicted.append(i[0])
    actual.append(i[1])

# Export results as CSV
from os import getcwd
currentDir = getcwd()
df = pd.DataFrame({'Predicted': predicted, 'Actual': actual})
csv = df.to_csv(currentDir + '/1-results.csv', index=False)

# Export correlations to prevent false positives/negatives as CSV
corr = pd.DataFrame(dataset.corr())
csv = corr.to_csv(currentDir + '/2-correlations.csv', index=False)

# Export coefficients that were used as CSV
coefficients = list(zip(dataset.iloc[:, :-1].columns, regressor.coef_))
coeff = pd.DataFrame(coefficients, columns=['Variable', 'Coefficient'])
csv = coeff.to_csv(currentDir + '/3-coefficients.csv', index=False)