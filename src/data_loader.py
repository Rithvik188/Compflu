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

def preprocess(dataset: Dict[float, Tuple[np.ndarray, np.ndarray]], floor: float = 1e-20,
               smooth_window: int = 41):
    """
    Cleans the raw trajectories for analysis.
    - Drops the last row (it has dt=0.05 instead of the uniform dt=0.5).
    - Works in y = log10(K); values below `floor` are numerical noise of the
      arrested state (the true floor is ~1.6e-26), so we cut there.
    - Returns a dict force -> dict(t, K, y, y_s, i_arr, t_arr, dt) where y_s is a
      centred moving average of y and i_arr is the first index with K < floor
      (len(t) if the trajectory never arrests).
    """
    from scipy.ndimage import uniform_filter1d

    out = {}
    for f in sorted(dataset.keys()):
        t, K = dataset[f]
        t, K = t[:-1], K[:-1]
        below = np.where(K < floor)[0]
        i_arr = int(below[0]) if len(below) else len(t)
        y = np.log10(np.clip(K, 1e-30, None))
        y_s = y.copy()
        y_s[:i_arr] = uniform_filter1d(y[:i_arr], smooth_window, mode='nearest')
        out[f] = dict(t=t, K=K, y=y, y_s=y_s, i_arr=i_arr,
                      t_arr=t[i_arr] if i_arr < len(t) else np.inf,
                      dt=float(t[1] - t[0]))
    return out

def load_clean(raw_dir: str = None, **kwargs):
    """Convenience: load + preprocess from data/raw."""
    if raw_dir is None:
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    return preprocess(load_and_preprocess_data(raw_dir), **kwargs)

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
