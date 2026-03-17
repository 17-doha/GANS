from keras.datasets.mnist import load_data
from matplotlib.pylab import randint
from app.visualizer import visualize_10
import numpy as np
from app.save_data import save


def generate_real_images(n_samples):
    real_imgs = X_train[randint(0, X_train.shape[0], n_samples)]
    real_imgs = real_imgs.reshape(real_imgs.shape[0], 28, 28, 1)
    y_real = np.ones((n_samples, 1))
    return real_imgs, y_real


def generate_fake_images(n_samples):
    fake_imgs = np.random.rand(28 * 28 * n_samples)
    fake_imgs = np.reshape(fake_imgs, (n_samples, 28, 28, 1))
    y_fake = np.zeros((n_samples, 1))
    return fake_imgs, y_fake


def generate_img_using_model(generator, noise_dim, n_samples):
    noise = np.random.randn(noise_dim * n_samples)
    noise = noise.reshape(n_samples, noise_dim)
    fake_imgs = generator.predict(noise)
    y_fake = np.zeros((n_samples, 1))
    return fake_imgs, y_fake


def generate_latent_points(noise_dim, batch_size):
    X_gan = np.random.randn(noise_dim * batch_size)
    X_gan = X_gan.reshape(batch_size, noise_dim)
    y_gan = np.ones((batch_size, 1))
    return X_gan, y_gan


(X_train, y_train), (X_test, y_test) = load_data()

print(
    "Shape of the training data",
    X_train.shape,
    ", the shape of the labels:",
    y_train.shape,
)
print(
    "Shape of the testing data",
    X_test.shape,
    ", the shape of the labels:",
    y_test.shape,
)


# Visualize the first 10 images in the training set
visualize_10(X_train)


# Reshape data to be 2D and normalize values to be between 0 and 1
X_train = np.reshape(X_train, (X_train.shape[0], 28 * 28))
X_test = np.reshape(X_test, (X_test.shape[0], 28 * 28))

X_train = X_train / 255.0
X_test = X_test / 255.0

X_real_imgs, y_real = generate_real_images(int(256 / 2))
X_fake_imgs, y_fake = generate_fake_images(int(256 / 2))

visualize_10(X_real_imgs)
visualize_10(X_fake_imgs)


X_gan = np.random.randn(100 * 10)
X_gan = X_gan.reshape(10, 100)

# Save the generated data
save(X_real_imgs, y_real, X_fake_imgs, y_fake, X_train, X_test, X_gan)
