'''Train and evaluate a fully connected network against the MNIST dataset.'''

# This is the first step in learning Keras
# I will be reproducing the code in the MNIST Keras fully connected network example
# so that the concepts of Keras make sense to me.

# There will be a set of hyper parameters that I will write at 
# the start, and I will experiment with running this in various
# configurations, including on FloydHub, and our clone of that
# which may involve using KubeFlow

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# Hyper parameters
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 20

def print_training_testing_dataset_shapes(x_train, y_train, x_test, y_test):
    '''Dump diagnostic information about the datasets'''
    def _format(text, tensor):
        return "%s: %s, type: %s" % (text, str(tensor.shape), tensor.dtype)

    print("Shape of datasets: ")
    print(_format("Training dataset", x_train))
    print(_format("Training results", y_train))
    print(_format("Testing dataset", x_test))
    print(_format("Testing results", y_test))

def normalize(tensor, target_type = 'float32'):
    '''This function converts tensor to the target type, and normalizes all of the
    numerical values to the range 0->1'''
    original_type = tensor.dtype
    tensor = tensor.astype(target_type)
    tensor /= np.iinfo(original_type).max
    return tensor

def import_training_and_test_data_and_reshape():
    '''Load the training and testing data from the built-in mnist data set
    wrappers in Keras, and convert them to floating point vectors of size
    (m, 28x28) where m is the number of examples in the training and test
    datasets. Also returns labels for the training and test datasets.'''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    assert x_train.dtype == 'uint8' and \
        y_train.dtype == 'uint8' and \
        x_test.dtype == 'uint8' and \
        y_test.dtype == 'uint8'

    # Flatten each 28x28 grayscale pixel grid into a single vector of length 28x28
    # Note that the final size of the training and test grids are (m, 28x28) where
    # m is the number of images in the training and test datasets.
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    # Convert all of the grayscale pixel data to float32 and normalize to range of 0-1
    # from a range of 0-255 max(uint8)
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # 1-hot encode the labels for train and test
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test

def create_model():
    '''Returns a fully connected model with 512 units in the first and second layers,
    relu activation, and softmax(10) activation in its final layer.'''
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28*28,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    return model

x_train, y_train, x_test, y_test = import_training_and_test_data_and_reshape()

# Print out the shapes of the training and testing examples and labels
print_training_testing_dataset_shapes(x_train, y_train, x_test, y_test)

model = create_model()

# Prints a summary of the model that we're using
model.summary()

# Compile the model for the underlying hardware implementation (i.e., CPU, GPU)
model.compile(loss='categorical_crossentropy', 
              optimizer=RMSprop(),
              metrics=['accuracy'])

# I wonder how fast this goes on truly fast hardware ... what is the rate limiting
# step on the mobile 940MX 2GB VRAM card that I'm using on my laptop?
print("Compiling ... er I mean Training ...")
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(x_test, y_test))

print("Evaluating:")
score = model.evaluate(x_test, y_test, verbose=0)

print("Results:")
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])