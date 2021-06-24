
import pandas as pd

# PURPOSE IS TO PREDICT PURCHASE RATES

# This example dataset assumes that "Country, Age and Salary"
# are the leading causes of Purchase Rate and has not gone
# through an optimized process regarding P-values or Adj. R-Squared

# Import the dataset
dataset = pd.read_csv('Data.csv')

# Below code assumes Depedent Variable is in the last column

# Independent Variables "Country, Age, Salary" values as array
x = dataset.iloc[:, :-1].values
# print(x)

# Dependent Variable "Purchased" values as array
y = dataset.iloc[:, 3].values
# print(y)

# Taking care of missing data as averages in the same column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
# print(x)

# Import categorical labeling libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
labelencoder_y = LabelEncoder()

# Encode the Independent Variable "Country" (France/Germany/Spain)
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
# .toarray() because there are more than 2 categories (France/Germany/Spain)
# print(x)

# Encode the Dependent Variable "Purchased" (Yes/No)
y = labelencoder_y.fit_transform(y)
# print(y)