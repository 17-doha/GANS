import os
import numpy as np

def check_data():
    # Construct path relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset", "gan_data.npz")
    
    if not os.path.exists(data_path):
        # Fallback to check if it's in the root or current dir
        data_path_root = os.path.join(os.getcwd(), "dataset", "gan_data.npz")
        if os.path.exists(data_path_root):
            data_path = data_path_root
        elif os.path.exists("gan_data.npz"):
            data_path = "gan_data.npz"
        else:
            print(f"Error: Data file not found at {data_path} or {data_path_root}")
            exit(1)
            
    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
        
    required_keys = [
        "X_real_imgs", "y_real", "X_fake_imgs", "y_fake", 
        "X_train_processed", "X_test_processed", "X_gan"
    ]
    
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        print(f"Error: Missing keys in data: {missing_keys}")
        exit(1)
        
    print("Schema Check Passed: All required keys are present.")
    
    # Optional: basic shape validations
    try:
        assert data["X_train_processed"].shape[1] == 28 * 28, "X_train_processed should be flattened 28x28 images"
        assert data["X_test_processed"].shape[1] == 28 * 28, "X_test_processed should be flattened 28x28 images"
        print("Data shape validation passed.")
    except AssertionError as e:
        print(f"Data validation failed: {e}")
        exit(1)
        
    print("Data validation successful!")
    
if __name__ == "__main__":
    check_data()
