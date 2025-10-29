from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt


def create_model(dim_output):
    "A first idea of the neural network's structure in order to compare different strengths."

    model = keras.models.Sequential()

    model.add(keras.Input(shape = (28, 28, 1)))    # Pictures are (28, 28) bitmaps

    model.add(layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'))
    model.add(layers.Conv2D(32, kernel_size = (3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D( pool_size = (2,2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(dim_output, activation = 'softmax'))     # Softmax guarantees a prob distribution as output

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'mse'])
    return model
    
def create_model_for_test(dim_output):
    """No more dropout layer."""
    model = keras.models.Sequential()
    model.add(keras.Input(shape = (28, 28, 1))) 
    model.add(layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'))
    model.add(layers.Conv2D(32, kernel_size = (3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D( pool_size = (2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(dim_output, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'mse'])
    return model

def create_fully_connected_model(num_cat = 5):
    """Assumes pca preprocessed data."""
    model = keras.models.Sequential()
    model.add(keras.Input(shape = (200,))) # the transformed images have 200 features
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(units= num_cat, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def create_fully_connected_model_for_train(num_cat = 5):
    """Assumes pca preprocessed data."""
    model = keras.models.Sequential()
    model.add(keras.Input(shape = (200,))) # the transformed images have 200 features
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units= num_cat, activation='softmax'))  # num_cat = 5, softmax guarantees a prob distribution as output
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def plot_history(history):
    x = range(history.params['epochs'])
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_title('accuracy')
    axarr[0].plot(x, acc, label='train')
    axarr[0].plot(x, val_acc, label='validation')
    axarr[0].legend()
    
    loss, val_loss = history.history['loss'], history.history['val_loss']
    axarr[1].set_title('loss')
    axarr[1].plot(x, loss, label='train')
    axarr[1].plot(x, val_loss, label='validation')
    axarr[1].legend()

    plt.show()
    return