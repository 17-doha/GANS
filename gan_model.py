import tensorflow as tf
from tensorflow.keras.models import Sequential
def building_gan(generator, discriminator):
    GAN = Sequential()
    discriminator.trainable = False
    # Adding the generator and the discriminator
    GAN.add(generator)
    GAN.add(discriminator)
    # Optimization function
    opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    # Compile the model 
    GAN.compile(loss='binary_crossentropy', optimizer=opt)
    return GAN