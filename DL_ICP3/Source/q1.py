from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values
print(np.unique(y))

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)
print("Done with tokenizing")

# Lebelencoding for converting text to numeric data
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
print("done with label encoding and training")

# Number of features
model = Sequential()
# 1) Input dimension is not defined
input_dim = 2000
model.add(layers.Dense(300,input_dim=input_dim, activation='relu'))
# Output labels are 3 as in the input provided we have only NEG, POS and UNSUP
# The activation will be softmax instead of sigmoid because sigmoid can be used for utmost 2 labels whereas
# softmax can be used for any number of labels. Here for the output we have three labels
model.add(layers.Dense(3, activation='softmax'))
# model.add(layers.Dense(3, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)
loss, accuracy = model.evaluate(X_test, y_test)
print("LOSS: {}".format(loss))
print("ACCURACY: {}".format(accuracy))

