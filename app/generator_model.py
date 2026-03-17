from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Conv2DTranspose, Reshape, LeakyReLU
from tensorflow.keras.models import Sequential


def building_generator(noise_dim):
    genModel = Sequential()
    genModel.add(Dense(128 * 6 * 6, input_dim=noise_dim))
    genModel.add(LeakyReLU())
    genModel.add(Reshape((6, 6, 128)))
    # Second layer
    genModel.add(Conv2DTranspose(128, (4, 4), strides=(2, 2)))
    genModel.add(LeakyReLU())
    # Third layer
    genModel.add(Conv2DTranspose(128, (4, 4), strides=(2, 2)))
    genModel.add(LeakyReLU())
    genModel.add(Conv2D(1, (3, 3), activation="sigmoid"))
    return genModel
