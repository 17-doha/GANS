
from data_generator import generate_img_using_model, generate_real_images, generate_latent_points
from discriminator_model import building_discriminator
import numpy as np
import matplotlib.pyplot as plt
import os


def training_gan(gan_model, discriminator, generator, batch_size=256, epochs=100, epoch_steps=468, noise_dim=100):
    half_batch = batch_size // 2 
    
    for epoch in range(0,epochs): 
        for step in range(0, epoch_steps):
         
            X_fake, y_fake = generate_img_using_model(generator, noise_dim, half_batch)
           
            X_real, y_real = generate_real_images(half_batch)
            
            # Creating training set (Total size is now 256, not 512)
            X_batch = np.concatenate([X_real, X_fake], axis = 0)
            y_batch = np.concatenate([y_real, y_fake], axis = 0)      
            
            discriminator.trainable = True
            d_loss, d_acc = discriminator.train_on_batch(X_batch, y_batch)
            
            # Generating noise input for the generator 
            X_gan, y_gan = generate_latent_points(noise_dim, batch_size)
            
            discriminator.trainable = False
            gan_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            # Report the progress
            report_porgress(epoch=epoch, step=step, d_loss=d_loss, gan_loss=gan_loss, noise_dim=noise_dim, epoch_steps=epoch_steps)
            
        # Report the progress on the full epoch
        report_porgress(epoch=epoch, step=step, d_loss=d_loss, gan_loss=gan_loss, noise_dim=noise_dim, epoch_steps=epoch_steps, generator=generator, discriminator=discriminator, eoe=True)


def report_porgress(epoch, step, d_loss, gan_loss, noise_dim = None, epoch_steps= None, generator=None, discriminator=None, n_samples=100, eoe= False):
    if eoe and step == (epoch_steps-1):
        # Report a full epoch training performance
        # Sample some real images from the training set
        X_real, y_real = generate_real_images(n_samples)
        # Measure the accuracy of the discrinminator on real sampled images
        _ , acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
        # Generates fake examples
        X_fake, y_fake = generate_img_using_model(generator, noise_dim, n_samples)
        # evaluate discriminator on fake images
        _, acc_fake = discriminator.evaluate(X_fake, y_fake, verbose=0)
        
        # 1. Ensure the output directory exists
        os.makedirs('output', exist_ok=True)
        
        # summarize discriminator performance
        # plot images
        for i in range(10 * 10):
            # define subplot
            plt.subplot(10, 10, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(X_fake[i, :, :, 0], cmap='gray_r')
            
        # 2. Update the filename to include the output/ folder path
        filename = 'output/generated_examples_epoch%04d.png' % (epoch+1)
        plt.savefig(filename)
        
        # 3. Clear the plot to prevent memory leaks across epochs
        plt.close()
        
        print('Disciminator Accuracy on real images: %.0f%%, on fake images: %.0f%%' % (acc_real*100, acc_fake*100))
        
        # save the generator model tile file
        filename = 'generator_model.h5'
        generator.save(filename)
    else:
        # Report a single step training performance 
        print('Training progress in epoch #%d, step %d, discriminator loss=%.3f , generator loss=%.3f' % (epoch, step ,d_loss, gan_loss))