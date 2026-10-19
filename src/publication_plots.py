import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from data_loader import load_and_preprocess_data
from hdmd_analysis import extract_relaxation_times
from deep_koopman import ForceConditionedKoopmanAE

rcParams['font.size'] = 14
rcParams['axes.labelsize'] = 16
rcParams['axes.titlesize'] = 18
rcParams['legend.fontsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['lines.linewidth'] = 2.0
rcParams['figure.figsize'] = (10, 6)
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

def plot_critical_slowing_down():
    forces, taus = extract_relaxation_times()
    plt.figure()
    plt.plot(forces, taus, marker='o', color='indigo', linestyle='-', markersize=8, label='HDMD Extracted $\\tau$')
    plt.axvline(1.65, color='crimson', linestyle='--', linewidth=2, label='Critical Force $f_c \\approx 1.65$')
    plt.yscale('log')
    plt.xlabel('Active Force ($f$)')
    plt.ylabel('Relaxation Time $\\tau$ (log scale)')
    plt.title('Critical Slowing Down near Jamming Threshold')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "pub_critical_slowing_down.png")
    plt.savefig(out_path)
    plt.close()

def plot_global_master_trajectory():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    dataset = load_and_preprocess_data(raw_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 100
    model = ForceConditionedKoopmanAE(input_dim=window_size, latent_dim=10, hidden_dim=64).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "koopman_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    forces = sorted(list(dataset.keys()))
    
    target_subset = {0.0, 0.5, 1.0, 1.4, 1.6, 1.8, 2.0}
    forces = [f for f in forces if f in target_subset]
    
    norm = mcolors.Normalize(vmin=min(forces), vmax=max(forces))
    cmap = cm.get_cmap('plasma')
    
    fig, ax = plt.subplots()
    
    for f in forces:
        time, ke = dataset[f]
        color = cmap(norm(f))
        
        # Ground truth
        ax.plot(time, ke, color=color, alpha=0.4, linestyle='-', linewidth=4.0)
        
        # Transform input to Log-Space to match the new model
        ke_log = np.log10(ke + 1e-12)
        
        predictions_log = []
        with torch.no_grad():
            f_tensor = torch.tensor([[f]], dtype=torch.float32).to(device)
            for i in range(len(ke_log) - window_size):
                x_c = torch.tensor(ke_log[i : i+window_size], dtype=torch.float32).unsqueeze(0).to(device)
                _, x_next_pred, _, _ = model(x_c, f_tensor)
                predictions_log.append(x_next_pred[0, -1].item())
                
        # Transform back to Linear-Space for plotting
        predictions_linear = 10 ** np.array(predictions_log) - 1e-12
        full_predictions = np.array([np.nan] * window_size + list(predictions_linear))
        
        safe_predictions = np.clip(full_predictions, a_min=1e-25, a_max=None)
        
        # Prediction
        ax.plot(time, safe_predictions, color=color, linestyle='--', linewidth=2.0, dashes=(5, 3), alpha=1.0)
        
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-10, top=1.0)
    ax.set_xlabel('Time')
    ax.set_ylabel('Kinetic Energy (log scale)')
    ax.set_title('Global Master Plot: ML Predictions vs Ground Truth')
    ax.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05)
    cbar.set_label('Active Force ($f$)', rotation=270, labelpad=20)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=4, alpha=0.4),
                    Line2D([0], [0], color='black', lw=2, linestyle='--', dashes=(5, 3))]
    ax.legend(custom_lines, ['Ground Truth Simulation', 'Deep Koopman Prediction'], loc='lower left')
    
    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "pub_master_trajectories.png")
    plt.savefig(out_path)
    print(f"Saved master plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_critical_slowing_down()
    plot_global_master_trajectory()
