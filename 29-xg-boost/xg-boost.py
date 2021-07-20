
import pandas as pd

# THE PURPOSE IS TO INCREASE ORIGINAL ACCURACY RATE

# Import the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset into Training & Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Apply XG Boost, Classifier import HARD CODED also has Regressor
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(cm)
print('')
print(score)
print('')

# Apply K-Fold Cross Validation
# "K" number of folds "cv" is HARD CODED
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print(f'K-Fold Accuracy {accuracies.mean()*100:.2f} %')
print('')
print(f'Standard Deviation {accuracies.std()*100:.2f} %')