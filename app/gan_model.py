import tensorflow as tf
from tensorflow.keras.models import Sequential


def building_gan(generator, discriminator):
    GAN = Sequential()
    GAN.add(generator)
    GAN.add(discriminator)
    opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    GAN.compile(loss="binary_crossentropy", optimizer=opt)
    return GAN
