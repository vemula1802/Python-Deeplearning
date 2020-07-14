from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from keras.layers.embeddings import Embedding
from keras.layers import Flatten


twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
y = twenty_train.target
sentences = twenty_train.data

max_review_len = max([len(s.split()) for s in sentences])


tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)

sentences = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

input_dim = 2000

# Build model
model = Sequential()
model.add(layers.Dense(300, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(20,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

# output
loss, accuracy = model.evaluate(X_test, y_test)
print("LOSS: {}".format(loss))
print("ACCURACY: {}".format(accuracy))


# Bonus part
# For accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# For loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
