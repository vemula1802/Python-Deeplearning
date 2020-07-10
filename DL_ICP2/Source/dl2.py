from keras import Sequential
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

print(train_images.shape[1:])
print(50*"++")
#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
print(dimData)
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /=255.0
test_data /=255.0
#change the labels frominteger to one-hot encoding. to_categorical is doing the same thing as LabelEncoder()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#creating network and Adding hidden layers
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))
loss, accuracy = model.evaluate(test_data, test_labels_one_hot)
print("The data with the initial model")
print("LOSS IS : {}".format(loss))
print("ACCURACY IS: {}".format(accuracy))

# 1 Plot loss and accuracy

print("Rendering Loss/Acc Trends...")
plt.figure()
plt.plot(np.arange(0, 10), history.history["val_loss"], label="validation loss")
plt.plot(np.arange(0, 10), history.history["val_accuracy"], label="validation accuracy")
plt.plot(np.arange(0, 10), history.history["loss"], label="training Loss")
plt.plot(np.arange(0, 10), history.history["accuracy"], label="training accuracy")
plt.title("the loss and accuracy for both training data and validation data")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best")
plt.show()

# 2 Single test image

print("Rendering test image...")
test_img_seven = test_images[27]
test_data_seven = test_data[[27], :]
plt.imshow(test_img_seven, cmap=plt.get_cmap('gray'))
plt.title("Model Prediction: {}".format(model.predict_classes(test_data_seven)[0]))
plt.show()

# 3 Change number of hidden layers and activation

print("Training a model with 2 more relu hidden layers...")
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print()
model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1,
          validation_data=(test_data, test_labels_one_hot))

loss2, accuracy2 = model.evaluate(test_data, test_labels_one_hot)
print("The data with more relu layers added to the existing model...")
print("LOSS IS {}: ".format(loss2))
print("ACCURACY IS {}: ".format(accuracy2))
print("NEW LOSS: {} CHANGE: {}".format(loss2, loss2 - loss))
print("NEW ACCURACY: {} CHANGE: {}".format(accuracy2, accuracy2 - accuracy))

# Training a model with sigmoid activation instead of relu
model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(dimData,)))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print()
model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1,
          validation_data=(test_data, test_labels_one_hot))

loss2, accuracy2 = model.evaluate(test_data, test_labels_one_hot)
print("Sigmoid model instead of relu")
print("LOSS IS {}: ".format(loss2))
print("ACCURACY IS {}: ".format(accuracy2))
print("NEW LOSS: {} CHANGE: {}".format(loss2, loss2 - loss))
print("NEW ACCURACY: {} CHANGE: {}".format(accuracy2, accuracy2 - accuracy))

# 4 Without scaling

# process the data
# 1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:784])
# print(dimData)
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')

# not scaling the data this time

# change the labels from integer to one-hot encoding. to_categorical is doing the same thing as LabelEncoder()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

print("Training a model with more relu hidden layers...")
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print()
model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1,
          validation_data=(test_data, test_labels_one_hot))

loss2, accuracy2 = model.evaluate(test_data, test_labels_one_hot)
print("Non-scaled model")
print("LOSS IS {}: ".format(loss2))
print("ACCURACY IS {}: ".format(accuracy2))
print("NEW LOSS: {} CHANGE: {}".format(loss2, loss2 - loss))
print("NEW ACCURACY: {} CHANGE: {}".format(accuracy2, accuracy2 - accuracy))