'''Training app for a Lenet-5 CNN on the MNIST dataset.'''

# This is the second step in learning Keras. The first
# was for a fully connected network. This time, I will be
# using the Lenet-5 model topology instead. Here, it should
# yield 0.9933% accuracy against the test dataset, with a 
# test loss of 0.04381.

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
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

    if keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
    
    # Flatten each 28x28 grayscale pixel grid into a single vector of length 28x28
    # Note that the final size of the training and test grids are (m, 28x28) where
    # m is the number of images in the training and test datasets.
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    # Convert all of the grayscale pixel data to float32 and normalize to range of 0-1
    # from a range of 0-255 max(uint8)
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # 1-hot encode the labels for train and test
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test, input_shape

def create_model(input_shape):
    '''Returns a convolutional neural network that uses the LeNet-5 topology.'''
    model = Sequential()

    # first set of CONV => RELU => POOL
    # we need to specify the input shape only for the first layer
    model.add(Conv2D(20, kernel_size=(5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # second set of CONV => RELU => POOL
    # in subsequent layers, Keras computes the input shape from the previous layer
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # third set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))

    # softmax classifier as final output
    model.add(Dense(10, activation="softmax"))

    return model

x_train, y_train, x_test, y_test, input_shape = import_training_and_test_data_and_reshape()

# Print out the shapes of the training and testing examples and labels
print_training_testing_dataset_shapes(x_train, y_train, x_test, y_test)

model = create_model(input_shape)

# Prints a summary of the model that we're using
model.summary()

# Compile the model for the underlying hardware implementation (i.e., CPU, GPU)
model.compile(loss='categorical_crossentropy', 
              optimizer=RMSprop(),
              metrics=['accuracy'])

# The rate limiting step here, after comparing training rates on 940MX vs. 965M 
# on my Surface Studio is clearly memory bandwidth. The 965M card has 2x the 
# bandwidth, and trains at 2x the rate.
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