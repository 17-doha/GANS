from matplotlib import pyplot as plt
import numpy as np
import torch

def visualize_10(X_data):
    fig, axs = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            img = X_data[i * 5 + j]
            # Handle PyTorch's (1, 28, 28) vs raw (28, 28)
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = img[:, :, 0]
                
            axs[i, j].imshow(img, cmap=plt.get_cmap("gray"))
            axs[i, j].axis("off")
    plt.show()


# Change the definition to match this exactly:
def plot_actual_vs_generated(generator, X_train, noise_dim, n_samples=10):
    from app.data_generator import generate_real_images, generate_img_using_model

    X_real, _ = generate_real_images(X_train, n_samples)
    X_fake, _ = generate_img_using_model(generator, noise_dim, n_samples)

    plt.figure(figsize=(10, 2.5))

    for i in range(n_samples):
        # REAL IMAGES
        plt.subplot(2, n_samples, 1 + i)
        plt.axis("off")
        # Handle PyTorch shape for Matplotlib
        real_img = X_real[i, 0] if X_real.shape[1] == 1 else X_real[i, :, :, 0]
        plt.imshow(real_img, cmap="gray_r")
        if i == n_samples // 2:
            plt.title("ACTUAL (REAL)")

        # FAKE IMAGES
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis("off")
        # Handle PyTorch shape for Matplotlib
        fake_img = X_fake[i, 0] if type(X_fake) == np.ndarray and X_fake.shape[1] == 1 else X_fake[i, 0].detach().cpu().numpy()
        plt.imshow(fake_img, cmap="gray_r")
        if i == n_samples // 2:
            plt.title("GENERATED (FAKE)")

    plt.tight_layout()
    plt.savefig("output/actual_vs_generated_comparison.png")
    plt.show()