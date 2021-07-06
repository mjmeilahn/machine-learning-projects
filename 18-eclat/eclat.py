
import pandas as pd

# THE PURPOSE IS TO FIND ASSOCIATIONS IN ITEMS BOUGHT TOGETHER

# Import the dataset
# This dataset does not have headers - specify "header=None"
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Since we do not have columns, loop through and create a list
# HARD CODED for 7501 & 20 rows
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
# print(transactions)

# Min Support & Confidence are HARD CODED from calculations
# min_support = 3 items in a transaction times 7 days / wk, divided by 7501 rows
# min_confidence = chosen for this example, play with the range .2 to .8
# min_lift = chosen as rule of thumb for all associations
# min_length, max_length = how many associations you want to see in results
from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=.2, min_lift=3, min_length=2, max_length=2)
results = list(rules)
# print(results)

# Prettify the Eclat Results
def inspect (results):
    leftHandSide = [tuple(result[2][0][0])[0] for result in results]
    rightHandSide = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(leftHandSide, rightHandSide, supports))

# Organize with a Data Frame
resultsInDataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

# Sort by Highest Support
associations = resultsInDataFrame.nlargest(n=10, columns='Support')

# Export results as CSV
from os import getcwd
currentDir = getcwd()
associations.to_csv(currentDir + '/1-results.csv', index=False)
