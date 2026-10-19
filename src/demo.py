"""
=============================================================
  DEMO: Decoding Dense Active Fluids Through Machine Learning
=============================================================
Run this script to reproduce ALL results end-to-end.

Usage:
    python src/demo.py              (uses pre-trained model)
    python src/demo.py --retrain    (retrains from scratch ~30 min)
"""
import os
import sys
import argparse
import numpy as np

# ── 0. Parse arguments ──────────────────────────────────────
parser = argparse.ArgumentParser(description="End-to-end demo of the Active Glass ML pipeline.")
parser.add_argument("--retrain", action="store_true", help="Retrain the model from scratch instead of using saved weights.")
args = parser.parse_args()

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
MODEL_PATH = os.path.join(ROOT, "data", "processed", "koopman_model.pth")
DOCS_DIR = os.path.join(ROOT, "docs")

print("=" * 60)
print("  Decoding Dense Active Fluids Through Machine Learning")
print("  End-to-End Demonstration")
print("=" * 60)

# ── 1. Data Overview ────────────────────────────────────────
print("\n[STEP 1/4] Loading raw simulation data...")
from data_loader import load_and_preprocess_data
dataset = load_and_preprocess_data(RAW_DIR)

forces = sorted(dataset.keys())
print(f"  Available active forces: {forces}")
print(f"  Total trajectories: {len(forces)}")

# Quick stats
for f in [forces[0], forces[len(forces)//2], forces[-1]]:
    t, ke = dataset[f]
    print(f"    f={f:.2f}: {len(t)} timesteps, KE range [{ke.min():.2e}, {ke.max():.2e}]")

# ── 2. Physics: Critical Slowing Down via HDMD ──────────────
print("\n[STEP 2/4] Extracting relaxation times via Hankel DMD...")
from hdmd_analysis import extract_relaxation_times
hdmd_forces, taus = extract_relaxation_times()

print("\n  Force  |  Relaxation Time (tau)")
print("  " + "-" * 35)
for f, tau in zip(hdmd_forces, taus):
    bar = "█" * min(int(tau / 200), 40)
    print(f"  {f:5.2f}  |  {tau:>8.1f}  {bar}")

# ── 3. ML Model ────────────────────────────────────────────
import torch
from deep_koopman import ForceConditionedKoopmanAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.retrain:
    print(f"\n[STEP 3/4] Retraining model from scratch on {device}...")
    print("  (This will take ~30 minutes on GPU)")
    from train import train_model
    train_model()
else:
    if os.path.exists(MODEL_PATH):
        print(f"\n[STEP 3/4] Loading pre-trained model from {MODEL_PATH}")
        print(f"  Device: {device}")
    else:
        print(f"\n[STEP 3/4] No pre-trained model found. Training from scratch on {device}...")
        from train import train_model
        train_model()

# Load model and run a quick zero-shot test
model = ForceConditionedKoopmanAE(input_dim=100, latent_dim=10, hidden_dim=64).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

test_forces = [0.5, 1.25, 1.6, 1.8]
print(f"\n  Zero-Shot Test on unseen forces: {test_forces}")

for tf in test_forces:
    if tf not in dataset:
        continue
    t, ke = dataset[tf]
    ke_log = np.log10(ke + 1e-12)
    
    preds = []
    with torch.no_grad():
        f_tensor = torch.tensor([[tf]], dtype=torch.float32).to(device)
        for i in range(min(500, len(ke_log) - 100)):
            x_c = torch.tensor(ke_log[i:i+100], dtype=torch.float32).unsqueeze(0).to(device)
            _, x_next_pred, _, _ = model(x_c, f_tensor)
            preds.append(x_next_pred[0, -1].item())
    
    true_vals = ke_log[100:100+len(preds)]
    mse = np.mean((np.array(preds) - true_vals) ** 2)
    print(f"    f={tf:.2f}  →  Log-Space MSE: {mse:.6e}  ✓")

# ── 4. Generate Publication Plots ───────────────────────────
print("\n[STEP 4/4] Generating publication-quality figures...")
from publication_plots import plot_critical_slowing_down, plot_global_master_trajectory

plot_critical_slowing_down()
print(f"  ✓ Saved: {os.path.join(DOCS_DIR, 'pub_critical_slowing_down.png')}")

plot_global_master_trajectory()
print(f"  ✓ Saved: {os.path.join(DOCS_DIR, 'pub_master_trajectories.png')}")

# ── Done ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO COMPLETE")
print("=" * 60)
print(f"\n  Output figures are in: {os.path.abspath(DOCS_DIR)}")
print("\n  To view them, open:")
print(f"    1. {os.path.join(DOCS_DIR, 'pub_critical_slowing_down.png')}")
print(f"    2. {os.path.join(DOCS_DIR, 'pub_master_trajectories.png')}")
print()
