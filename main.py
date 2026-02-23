
from numpy.random import randint
import os
import numpy as np
from generator_model import building_generator
from discriminator_model import building_discriminator
from gan_model import building_gan
from trainer import training_gan
import tensorflow as tf
from visualizer import plot_actual_vs_generated, visualize_10
np.random.seed(42)
data = np.load('dataset/gan_data.npz')

# Extract the arrays using the names you gave them when saving
X_real_imgs = data['X_real_imgs']
y_real = data['y_real']
X_fake_imgs = data['X_fake_imgs']
y_fake = data['y_fake']
X_train = data['X_train_processed']
X_test = data['X_test_processed']
X_gan = data['X_gan']

# They now have the exact same matrices with the exact same shapes and floating values!
print("Loaded real images shape:", X_real_imgs.shape)
print("Loaded fake images shape:", X_fake_imgs.shape)
# Loading tensorflow related libraries 
epochs = 10

batch_size = 256 
epoch_steps = int((2 * X_train.shape[0]/batch_size)/2)
print(epoch_steps)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

noise_dim = 100
generator = building_generator(noise_dim)
discriminator = building_discriminator()
gan_model = building_gan(generator, discriminator)
training_gan(gan_model, discriminator, generator, batch_size=batch_size, epochs=epochs, epoch_steps=epoch_steps, noise_dim=noise_dim)


model = tf.keras.models.load_model('generator_model.h5')
X_gan = model.predict(X_gan)

print("Generated images shape:", X_gan.shape)

visualize_10(X_gan)
plot_actual_vs_generated(model, noise_dim=100, n_samples=10)