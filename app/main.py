import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow

# Assuming your imports from app still exist and are converted to PyTorch
from generator_model import building_generator
from discriminator_model import building_discriminator
from trainer import training_gan
from visualizer import plot_actual_vs_generated, visualize_10

np.random.seed(42)
torch.manual_seed(42)

# Determine device (Use GPU if available for PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = np.load("dataset/gan_data.npz")
X_train = data["X_train_processed"]
X_gan = data["X_gan"]
X_train = X_train.reshape(-1, 1, 28, 28)
# X_gan = X_gan.reshape(-1, 1, 28, 28)


epochs = 5
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

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Assignment3_DohaHemdan")

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
plot_actual_vs_generated(generator, noise_dim=100, n_samples=10)