import pandas as pd
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn import preprocessing
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.utils.np_utils import to_categorical

# Read file
train = pd.read_csv("train.tsv", sep="\t")

# Assign Value
X = train['Phrase'].values
y = train['Sentiment'].values

# Tokenizing
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)

# Encoding
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)

# CNN layers
model = Sequential()
model.add(Embedding(2000, X.shape[1]))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, verbose=True, validation_data=(x_test, y_test), batch_size=256)

# Accuracy score
scores = model.evaluate(x_test, y_test)
print("Model accuracy: " + str((scores[1]*100)))

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()
