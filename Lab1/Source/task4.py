# importing libraries
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

spam_data = pd.read_csv('spam.csv', encoding='latin-1')
print(spam_data)
print("*"*50)

spam_data = spam_data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
print(spam_data)
print("*"*50)

# Count Vectorizer vs. Tfidf

# Tfidf
from sklearn.model_selection import train_test_split
vect = TfidfVectorizer()
spam_data1 = vect.fit_transform(spam_data.Text)
X_train, X_test, y_train, y_test = train_test_split(spam_data1,spam_data['Class'], test_size=0.2)
print(X_train)
print(50*"++")
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
mnnb = MultinomialNB()
mnnb.fit(X_train,y_train)
feature_names = np.array(vect.get_feature_names())
y_pred = mnnb.predict(X_test)
print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))

# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
spam_data2 = vect.fit_transform(spam_data.Text)
X_train, X_test, y_train, y_test = train_test_split(spam_data2,spam_data['Class'], test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)
feature_names = np.array(vect.get_feature_names())
y_pred = model.predict(X_test)
print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))