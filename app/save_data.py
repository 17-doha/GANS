import os
import numpy as np

def save(X_real_imgs, y_real, X_fake_imgs, y_fake, X_train, X_test, X_gan):
    save_dir = "dataset"
    os.makedirs(save_dir, exist_ok=True)

    # 2. Save the arrays into a compressed .npz file
    save_path = os.path.join(save_dir, "gan_data.npz")
    np.savez_compressed(
        save_path, 
        X_real_imgs=X_real_imgs, 
        y_real=y_real,
        X_fake_imgs=X_fake_imgs, 
        y_fake=y_fake,
        X_train_processed=X_train, 
        X_test_processed=X_test,
        X_gan=X_gan  
    )

    print(f"Data successfully saved to {save_path}")