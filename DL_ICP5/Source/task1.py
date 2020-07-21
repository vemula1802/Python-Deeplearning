from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np

myString = ["A lot of good things are happening. We are respected again throughout "
            "the world, and that's a great thing.@realDonaldTrump"]

savedmodel = load_model('sentimentmodel.h5')

# vectorizing the tweet by the pre-fitted tokenizer instance
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(myString)
myString = tokenizer.texts_to_sequences(myString )

# padding the tweet to have exactly the same shape as `embedding_2` input
myString = pad_sequences(myString, maxlen=28, dtype='int32', value=0)
predictionResult = savedmodel.predict_classes(myString)

print("Predicted Class is: ", predictionResult)
sentiment = savedmodel.predict(myString, batch_size=1, verbose=2)[0]
print(sentiment)
if(np.argmax(sentiment) == 1):
    print(" The Statement is Negative")
elif (np.argmax(sentiment) == 0):
    print(" The Statement is Positive")
elif():
    print(" The Statement is Neutral")
