from matplotlib import pyplot as plt

def visualize_10(X_train):
    fig, axs = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):  
            axs[i,j].imshow(X_train[i+j], cmap=plt.get_cmap('gray'))

    plt.show()


def plot_actual_vs_generated(generator, noise_dim, n_samples=10):
    from app.data_generator import generate_img_using_model, generate_real_images
    
    X_real, _ =     generate_real_images(n_samples)

    X_fake, _ = generate_img_using_model(generator, noise_dim, n_samples)
    
    plt.figure(figsize=(10, 2.5))
    
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_real[i, :, :, 0], cmap='gray_r')
        if i == 5: 
            plt.title("ACTUAL (REAL)")
            
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fake[i, :, :, 0], cmap='gray_r')
        if i == 5: 
            plt.title("GENERATED (FAKE)")
            
    plt.tight_layout()
    
    plt.savefig('output/actual_vs_generated_comparison.png')
    plt.show()