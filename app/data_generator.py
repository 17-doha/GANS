import numpy as np
import torch
from torchvision import datasets
from app.save_data import save


def generate_real_images(X_train, n_samples):
    idx = np.random.randint(0, X_train.shape[0], n_samples)
    real_imgs = X_train[idx]
    # Reshaping to PyTorch standard: (Batch, Channels, Height, Width)
    real_imgs = real_imgs.reshape(n_samples, 1, 28, 28)
    y_real = np.ones((n_samples, 1))
    return real_imgs, y_real


def generate_fake_images(n_samples):
    fake_imgs = np.random.rand(n_samples, 1, 28, 28)
    y_fake = np.zeros((n_samples, 1))
    return fake_imgs, y_fake


def generate_img_using_model(generator, noise_dim, n_samples):
    noise = torch.randn(n_samples, noise_dim)
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(noise).cpu().numpy()
    generator.train()
    y_fake = np.zeros((n_samples, 1))
    return fake_imgs, y_fake


def generate_latent_points(noise_dim, batch_size):
    X_gan = np.random.randn(batch_size, noise_dim)
    y_gan = np.ones((batch_size, 1))
    return X_gan, y_gan


if __name__ == "__main__":
    print("Downloading MNIST using PyTorch...")
    # Load MNIST using PyTorch instead of Keras
    train_data = datasets.MNIST(root="./data", train=True, download=True)
    test_data = datasets.MNIST(root="./data", train=False, download=True)

    X_train_raw = train_data.data.numpy()
    X_test_raw = test_data.data.numpy()

    print("Shape of the training data:", X_train_raw.shape)

    # Flatten and normalize data
    X_train = np.reshape(X_train_raw, (X_train_raw.shape[0], 28 * 28))
    X_test = np.reshape(X_test_raw, (X_test_raw.shape[0], 28 * 28))

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_real_imgs, y_real = generate_real_images(X_train, 128)
    X_fake_imgs, y_fake = generate_fake_images(128)

    X_gan = np.random.randn(10, 100)

    # Save the generated data
    save(X_real_imgs, y_real, X_fake_imgs, y_fake, X_train, X_test, X_gan)
