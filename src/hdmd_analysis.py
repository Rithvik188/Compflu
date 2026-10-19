import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data
from hdmd_core import KoopmanHDMD

def extract_relaxation_times():
    """
    Runs HDMD on all trajectories to extract the dominant relaxation time tau.
    """
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    dataset = load_and_preprocess_data(raw_dir)
    
    # We only care about jamming regime for critical slowing down
    # fc is around 1.65, so we analyze forces < 1.65
    forces = []
    taus = []
    
    print("Extracting eigenvalues via HDMD...")
    # Initialize HDMD model
    # We'll use a delay window of 1000 for these long trajectories
    # Rank of 10 should capture the dominant dynamics
    model = KoopmanHDMD(delay_window=1000, rank=10, use_gpu=True)
    
    for f in sorted(dataset.keys()):
        if f >= 1.65:
            continue
            
        time, ke = dataset[f]
        
        # We need the signal to decay to 0, which happens in the jamming regime.
        # Run HDMD on the kinetic energy signal
        eigs = model.fit(ke)
        
        # Eigenvalues in discrete time: lambda = exp(dt * continuous_lambda)
        # We are looking for the slowest decaying mode (closest to 1.0 magnitude, but < 1.0)
        # Note: Since the signal is real and decaying, the dominant eigenvalue is usually purely real.
        mags = np.abs(eigs)
        
        # Filter for eigenvalues inside the unit circle (stable decay)
        valid_eigs = eigs[(mags < 0.9999) & (mags > 0.0)]
        
        if len(valid_eigs) == 0:
            print(f"Force {f}: No stable decaying modes found.")
            continue
            
        # The dominant mode is the one with magnitude closest to 1 (slowest decay)
        # tau = -1 / ln(|lambda|)
        valid_mags = np.abs(valid_eigs)
        dominant_lambda = valid_mags[np.argmax(valid_mags)]
        tau = -1.0 / np.log(dominant_lambda)
        
        print(f"Force {f}: Dominant Eigenvalue = {dominant_lambda:.6f}, Relaxation Time (tau) = {tau:.2f}")
        
        forces.append(f)
        taus.append(tau)
        
    return np.array(forces), np.array(taus)

if __name__ == "__main__":
    forces, taus = extract_relaxation_times()
    
    # Plot tau vs force
    plt.figure(figsize=(8, 6))
    plt.plot(forces, taus, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Active Force ($f$)', fontsize=14)
    plt.ylabel('Relaxation Time $\\tau$ (timesteps)', fontsize=14)
    plt.title('Critical Slowing Down: Relaxation Time vs Active Force', fontsize=16)
    plt.grid(True)
    plt.axvline(1.65, color='r', linestyle='--', label='Critical Force $f_c \\approx 1.65$')
    plt.legend()
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    plt.savefig(os.path.join(output_dir, "relaxation_time_scaling.png"), dpi=300, bbox_inches='tight')
    print("Saved plot to docs/relaxation_time_scaling.png")
