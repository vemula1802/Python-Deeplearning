from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
#Adding hidden layer on both sides
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)

# "decoded" is the lossy reconstruction of the input
output_img = Dense(784, activation='sigmoid')(decoded)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, output_img)

# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
from keras.datasets import mnist, fashion_mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

history = autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


 # Q2) image before reconstruction
plt.imshow(x_test[2].reshape(28,28))
plt.title("Image to Be Encoded")
plt.show()

prediction = autoencoder.predict(x_test[2].reshape(1,784))

# Image after reconstruction
plt.imshow(prediction.reshape(28,28))
plt.title("Image Decoded")
plt.show()

# Bonus point
encoder = Model(input_img, encoded)
prediction = encoder.predict(x_test[2].reshape(1,784))
plt.imshow(prediction.reshape(16,2))

plt.title("Image after encoding")
plt.show()



# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()