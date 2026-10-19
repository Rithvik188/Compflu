import os
import glob
import numpy as np
from typing import Tuple, Dict

def load_and_preprocess_data(raw_dir: str):
    """
    Parses the raw avg_ke_1000_{force} files.
    Returns a dictionary mapping active force -> (time, kinetic_energy).
    """
    file_pattern = os.path.join(raw_dir, "avg_ke_1000_*")
    files = glob.glob(file_pattern)
    
    dataset = {}
    for fpath in files:
        # Extract active force from filename
        basename = os.path.basename(fpath)
        try:
            force_str = basename.split("1000_")[-1]
            force = float(force_str)
        except ValueError:
            print(f"Skipping {basename} - could not parse force.")
            continue
            
        # Load data (time, KE, discard 3rd column)
        data = np.loadtxt(fpath)
        time = data[:, 0]
        ke = data[:, 1]
        
        dataset[force] = (time, ke)
        
    print(f"Loaded {len(dataset)} trajectories.")
    return dataset

def create_train_test_split(dataset: Dict[float, Tuple[np.ndarray, np.ndarray]], test_forces: list):
    """
    Splits the dataset into training and testing sets based on held-out forces.
    """
    train_data = {}
    test_data = {}
    
    for f, (t, ke) in dataset.items():
        if f in test_forces:
            test_data[f] = (t, ke)
        else:
            train_data[f] = (t, ke)
            
    return train_data, test_data

if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    
    dataset = load_and_preprocess_data(raw_dir)
    
    # We want to predict for f < 1.65. 
    # Let's hold out some forces strictly below 1.65 for testing the ML model.
    # We will pick a few spread-out values.
    available_forces = sorted(list(dataset.keys()))
    print(f"Available forces: {available_forces}")
    
    test_forces = [0.5, 1.25, 1.5]
    print(f"Holding out test forces: {test_forces}")
    
    train, test = create_train_test_split(dataset, test_forces)
    print(f"Train size: {len(train)}, Test size: {len(test)}")
