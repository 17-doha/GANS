import os
import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow

from app.generator_model import building_generator
from app.discriminator_model import building_discriminator
from app.trainer import training_gan
from app.visualizer import plot_actual_vs_generated, visualize_10

# 1. Create the logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

try:
    np.random.seed(42)
    torch.manual_seed(42)

    # Determine device (Use GPU if available for PyTorch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load("dataset/gan_data.npz")
    X_train_np = data["X_train_processed"]

    X_train_np = X_train_np.reshape(-1, 1, 28, 28)

    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)

    X_gan = data["X_gan"]

    epochs = 15
    batch_size = 256
    epoch_steps = int((2 * X_train.shape[0] / batch_size) / 2)
    learning_rate = 2e-4
    noise_dim = 100

    # Initialize PyTorch Models
    generator = building_generator(noise_dim).to(device)
    discriminator = building_discriminator().to(device)

    # Initialize PyTorch Optimizers and Loss Function
    # PyTorch handles learning rate inside the optimizer, not the model definition
    opt_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    mlflow.set_tracking_uri("https://dagshub.com/17-doha/GANS.mlflow")
        
    # Set a NEW experiment name to bypass the DagsHub cache bug
    mlflow.set_experiment("GAN_Deployment_Run")

    training_gan(
        generator=generator, 
        discriminator=discriminator, 
        opt_g=opt_g, 
        opt_d=opt_d, 
        criterion=criterion,
        X_train=X_train,
        batch_size=batch_size,
        epochs=epochs,
        epoch_steps=epoch_steps,
        noise_dim=noise_dim,
        lr=learning_rate,
        device=device
    )

    # --- Post Training Generation ---
    # Load the saved PyTorch weights
    generator.load_state_dict(torch.load("generator_model.pt"))
    generator.eval() # Set to evaluation mode

    # Generate final images
    with torch.no_grad():
        # Convert X_gan to tensor if it's the latent space input
        latent_points = torch.tensor(X_gan, dtype=torch.float32).to(device)
        generated_tensors = generator(latent_points)
        generated_images = generated_tensors.cpu().numpy()

    print("Generated images shape:", generated_images.shape)

    visualize_10(generated_images)
    # Pass X_train to the visualizer on the last line
    plot_actual_vs_generated(generator, X_train, 100, 10)

except Exception as e:
    # 2. Catch the error, format the stack trace, and save it to the file
    error_message = traceback.format_exc()
    print("An error occurred! Writing to logs/error_logs.txt...")
    print(error_message)
    
    with open("logs/error_logs.txt", "w") as log_file:
        log_file.write(error_message)
        
    # 3. Exit with a non-zero status code so GitHub Actions registers the failure
    sys.exit(1)