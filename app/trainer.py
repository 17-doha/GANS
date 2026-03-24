import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mlflow

def calculate_accuracy(predictions, labels):
    """Helper function to replace Keras's model.evaluate()"""
    preds_rounded = (predictions >= 0.5).float()
    correct = (preds_rounded == labels).float().sum()
    return (correct / len(labels)).item()

def report_progress(
    epoch, step, d_loss, g_loss, generator, discriminator, criterion, 
    X_train, noise_dim, epoch_steps, n_samples=100, eoe=False, device="cpu"
):
    if eoe and step == (epoch_steps - 1):
        # --- Evaluate Full Epoch ---
        generator.eval()
        discriminator.eval()
        
        with torch.no_grad():
            # 1. Real Images Accuracy
            idx = torch.randint(0, X_train.shape[0], (n_samples,), device=device)
            X_real = X_train[idx]
            y_real = torch.ones((n_samples, 1), device=device)
            
            pred_real = discriminator(X_real)
            acc_real = calculate_accuracy(pred_real, y_real)

            # 2. Fake Images Accuracy
            noise = torch.randn(n_samples, noise_dim).to(device)
            X_fake = generator(noise)
            y_fake = torch.zeros((n_samples, 1)).to(device)
            
            pred_fake = discriminator(X_fake)
            acc_fake = calculate_accuracy(pred_fake, y_fake)

        os.makedirs("output", exist_ok=True)

        # Plot images (Convert PyTorch tensors back to Numpy for matplotlib)
        X_fake_np = X_fake.cpu().numpy()
        for i in range(10 * 10):
            plt.subplot(10, 10, 1 + i)
            plt.axis("off")
            plt.imshow(X_fake_np[i, :, :, 0] if X_fake_np.ndim == 4 else X_fake_np[i, 0, :, :], cmap="gray_r")

        filename = f"output/generated_examples_epoch{epoch + 1:04d}.png"
        plt.savefig(filename)
        plt.close()

        print(f"Discriminator Accuracy on real: {acc_real * 100:.0f}%, on fake: {acc_fake * 100:.0f}%")

        # LOG ACCURACY & IMAGES TO MLFLOW
        mlflow.log_metric("accuracy_real", acc_real, step=epoch)
        mlflow.log_metric("accuracy_fake", acc_fake, step=epoch)
        mlflow.log_artifact(filename, artifact_path="generated_images")

        # Save PyTorch Model weights
        torch.save(generator.state_dict(), "generator_model.pt")
        
        generator.train()
        discriminator.train()
    else:
        if step % 50 == 0:
            print(f"Training progress in epoch #{epoch}, step {step}, discriminator loss={d_loss:.3f} , generator loss={g_loss:.3f}")

def training_gan(
    generator, discriminator, opt_g, opt_d, criterion, X_train,
    batch_size=256, epochs=100, epoch_steps=468, noise_dim=100, lr=0.001, device="cpu"
):
    half_batch = batch_size // 2
    mlflow.set_tracking_uri("https://dagshub.com/17-doha/GANS.mlflow")
    with mlflow.start_run() as run:
        mlflow.set_tag("student_id", "202200701")
        
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("noise_dim", noise_dim)

        generator.to(device)
        discriminator.to(device)

        for epoch in range(epochs):
            for step in range(epoch_steps):
                
                # ==========================================
                # 1. Train Discriminator
                # ==========================================
                opt_d.zero_grad()
                
                # Real data
                idx = torch.randint(0, X_train.shape[0], (half_batch,), device=device)
                X_real = X_train[idx]
                y_real = torch.ones((half_batch, 1), device=device)
                
                d_real_loss = criterion(discriminator(X_real), y_real)

                # Fake data
                noise = torch.randn(half_batch, noise_dim).to(device)
                X_fake = generator(noise)
                y_fake = torch.zeros((half_batch, 1)).to(device)
                
                # Use .detach() so we don't backpropagate through the Generator
                d_fake_loss = criterion(discriminator(X_fake.detach()), y_fake)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                opt_d.step()

                # ==========================================
                # 2. Train Generator
                # ==========================================
                opt_g.zero_grad()
                
                noise_g = torch.randn(batch_size, noise_dim).to(device)
                X_gan = generator(noise_g)
                # Generator wants the Discriminator to think the fake images are REAL (1)
                y_gan = torch.ones((batch_size, 1)).to(device) 
                
                g_loss = criterion(discriminator(X_gan), y_gan)
                g_loss.backward()
                opt_g.step()

                # Report progress
                report_progress(
                    epoch, step, d_loss.item(), g_loss.item(), generator, discriminator, 
                    criterion, X_train, noise_dim, epoch_steps, device=device
                )

            # --- Log metrics at end of Epoch ---
            mlflow.log_metric("discriminator_loss", d_loss.item(), step=epoch)
            mlflow.log_metric("generator_loss", g_loss.item(), step=epoch)

            # Trigger end-of-epoch reports
            report_progress(
                epoch, step, d_loss.item(), g_loss.item(), generator, discriminator, 
                criterion, X_train, noise_dim, epoch_steps, eoe=True, device=device
            )
            
            # Save PyTorch model to MLflow
            if epoch % 10 == 0 or epoch == epochs - 1:
                mlflow.pytorch.log_model(generator, f"generator-epoch-{epoch}")
        
        run_id = run.info.run_id
        with open("model_info.txt", "w") as f:
             f.write(run_id)