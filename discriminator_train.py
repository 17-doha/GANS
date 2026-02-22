

from data_generator import generate_real_images, generate_fake_images
from discriminator_model import building_discriminator

discriminator = building_discriminator()

def training_discriminator(discriminator, epochs=100, n_batch=256):
    for i in range(epochs):
        # Generate true samples 
        X_real_imgs, y_real = generate_real_images(int(n_batch/2))
        # train the model on a collected batch
        _, acc_on_real = discriminator.train_on_batch(X_real_imgs, y_real)
        # Generate fake samples
        X_fake_imgs, y_fake = generate_fake_images(int(n_batch/2))
        # train the model on a collected batch
        _, acc_on_fake = discriminator.train_on_batch(X_fake_imgs, y_fake)
        # Display training performance
        print('Accuracy in epoch %d, on real images=%.0f%% , on fake images=%.0f%%' % (i+1, acc_on_real * 100, acc_on_fake * 100))
