from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print("MultinomialNB: " + str(score))

# TO SVC
from sklearn.svm import SVC
clf1 = SVC(kernel='linear')
clf1.fit(X_train_tfidf, twenty_train.target)
predicted = clf1.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print("SVC: " + str(score))


#using bigrams
tfidf_Vect_bi = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf_bi = tfidf_Vect_bi.fit_transform(twenty_train.data)
clf_bi = MultinomialNB()
clf_bi.fit(X_train_tfidf_bi, twenty_train.target)
X_test_tfidf_bi = tfidf_Vect_bi.transform(twenty_test.data)
predicted_bi = clf_bi.predict(X_test_tfidf_bi)
score_bi = metrics.accuracy_score(twenty_test.target, predicted_bi)
print("MultinomialNB + Bigram:" + str(score_bi))


# with stop words = english
tfidf_Vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("MultinomialNB + Bigram + Stopword: " + str(score))

