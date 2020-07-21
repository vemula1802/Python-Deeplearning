import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Sentiment.csv')# Keeping only the neccessary columns input as text and target as sentiment
data = data[['text','sentiment']]
# To return the lower-cased string from the given string.
# It converts all uppercase characters to lowercase.
data['text'] = data['text'].apply(lambda x: x.lower())
# Checks whether other than these chars are present in the data
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))


for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)

embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

#Converting the text data into numeric
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
print(integer_encoded)
print(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

#Applying GridSearchCV on the given dataset
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=createmodel,verbose=2)
batch_size= [10, 20]
epochs = [1, 2]
param_grid= {'batch_size':batch_size, 'epochs':epochs}
from sklearn.model_selection import GridSearchCV
grid  = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result= grid.fit(X_train, y=Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))