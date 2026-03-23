from keras.datasets.mnist import load_data
from matplotlib.pylab import randint
from visualizer import visualize_10
import numpy as np
import torch
from save_data import save

def generate_real_images(n_samples):
    real_imgs = X_train[randint(0, X_train.shape[0], n_samples)]
    # Reshaping to PyTorch standard: (Batch, Channels, Height, Width)
    real_imgs = real_imgs.reshape(real_imgs.shape[0], 1, 28, 28)
    y_real = np.ones((n_samples, 1))
    return real_imgs, y_real

def generate_fake_images(n_samples):
    fake_imgs = np.random.rand(n_samples, 1, 28, 28)
    y_fake = np.zeros((n_samples, 1))
    return fake_imgs, y_fake

def generate_img_using_model(generator, noise_dim, n_samples):
    noise = torch.randn(n_samples, noise_dim)
    
    # Use PyTorch evaluation mode
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(noise).cpu().numpy()
    generator.train() # Set back to train mode just in case
    
    y_fake = np.zeros((n_samples, 1))
    return fake_imgs, y_fake

def generate_latent_points(noise_dim, batch_size):
    X_gan = np.random.randn(batch_size, noise_dim)
    y_gan = np.ones((batch_size, 1))
    return X_gan, y_gan

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    (X_train_raw, y_train), (X_test_raw, y_test) = load_data()

    print("Shape of the training data", X_train_raw.shape)

    # Visualize the first 10 images
    visualize_10(X_train_raw)

    # Flatten and normalize data
    X_train = np.reshape(X_train_raw, (X_train_raw.shape[0], 28 * 28))
    X_test = np.reshape(X_test_raw, (X_test_raw.shape[0], 28 * 28))

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_real_imgs, y_real = generate_real_images(128)
    X_fake_imgs, y_fake = generate_fake_images(128)

    X_gan = np.random.randn(10, 100)

    # Save the generated data
    save(X_real_imgs, y_real, X_fake_imgs, y_fake, X_train, X_test, X_gan)