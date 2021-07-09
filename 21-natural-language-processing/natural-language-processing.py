
import pandas as pd

# THE PURPOSE IS TO PREDICT A RESPONSE BASED ON LEARNING TEXT

# Import data, specify TSV, ignore quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
text = dataset.iloc[:, 0].values
# print(text)

# Cleaning and ETL, ignore irrelevant words and conjugations
import re
import nltk # for natural language processing
nltk.download('stopwords')
from nltk.corpus import stopwords # for irrelevant words
from nltk.stem.porter import PorterStemmer # for conjugations

collection = []

# HARD CODED RANGE
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', text[i]) # Regex sub
    review = review.lower() # lowercase
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    collection.append(review)
# print(collection)

# Set Maximum Words Before Data Split
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # rounded to 1500 from len(x[0])
x = cv.fit_transform(collection).toarray()
y = dataset.iloc[:, -1].values
# print(len(x[0])) # 1566

# Split Data into Train & Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# OPTIONAL: Picking Naive Bayes Model for Prediction
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predict Test Model Results
import numpy as np
y_pred = classifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Make The Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
