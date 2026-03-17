## Building Discriminative Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential


def building_discriminator():
    # The image dimensions provided as inputs
    image_shape = (28, 28, 1)
    disModel = Sequential()
    disModel.add(Conv2D(64, 3, strides=2, input_shape=image_shape))
    disModel.add(LeakyReLU())
    disModel.add(Dropout(0.4))
    # Second layer
    disModel.add(Conv2D(64, 3, strides=2))
    disModel.add(LeakyReLU())
    disModel.add(Dropout(0.4))
    # Flatten the output
    disModel.add(Flatten())
    disModel.add(Dense(1, activation="sigmoid"))
    # Optimization function
    opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    # Compile the model
    disModel.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return disModel
