# GANs Training Pipeline

A robust, containerized machine learning project for training Generative Adversarial Networks (GANs). This repository includes the model architecture, training loop, visualization tools, and a fully automated CI/CD pipeline using GitHub Actions.

## 📁 Project Structure

The core machine learning logic is contained within the `app/` directory:

- **`main.py`**: The main entry point to initialize and start the training process.
- **`generator_model.py` & `discriminator_model.py`**: TensorFlow/Keras definitions for the generator and discriminator networks.
- **`gan_model.py`**: The combined GAN architecture.
- **`trainer.py`**: Handles the custom training loop, managing epochs, batches, and loss calculations.
- **`data_generator.py`**: Handles loading, structuring, and preprocessing the training dataset.
- **`visualizer.py` & `save_data.py`**: Utilities for plotting generated images, graphing loss curves, and exporting trained model weights.
- **`output/` & `results/`**: Output directories where generated examples and training performance graphs are automatically saved during training.

## 🚀 Getting Started

### Option 1: Running with Docker (Recommended)

This project is fully containerized. You can build and run the training pipeline in an isolated environment without installing local dependencies.

1. Build the Docker image:
   ```bash
   docker build -t gans-app .
   ```
