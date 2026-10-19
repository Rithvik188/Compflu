import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from data_loader import load_and_preprocess_data, create_train_test_split
from deep_koopman import ForceConditionedKoopmanAE

def prepare_sliding_windows(dataset_dict, window_size=50):
    X_curr, X_next, Forces = [], [], []
    for f, (time, ke) in dataset_dict.items():
        # FIX: Log-transform the kinetic energy. 
        # Standard MSE loss completely ignores the tiny tail values (1e-5), 
        # causing the neural network's ReLUs to lazily snap to zero early.
        # Log scaling forces the network to learn the tail dynamics just as accurately as the initial drop!
        ke_log = np.log10(ke + 1e-12)
        N = len(ke_log)
        for i in range(N - window_size):
            X_curr.append(ke_log[i : i + window_size])
            X_next.append(ke_log[i + 1 : i + window_size + 1])
            Forces.append([f])
    X_curr = torch.tensor(np.array(X_curr), dtype=torch.float32)
    X_next = torch.tensor(np.array(X_next), dtype=torch.float32)
    Forces = torch.tensor(np.array(Forces), dtype=torch.float32)
    return X_curr, X_next, Forces

def train_model():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    dataset = load_and_preprocess_data(raw_dir)
    
    test_forces = [0.5, 1.25, 1.6, 1.8]
    train_data, test_data = create_train_test_split(dataset, test_forces)
    
    window_size = 100
    
    print("Preparing sliding windows in Log Space...")
    X_train_c, X_train_n, F_train = prepare_sliding_windows(train_data, window_size)
    X_test_c, X_test_n, F_test = prepare_sliding_windows(test_data, window_size)
    
    print(f"Training samples: {len(X_train_c)}, Testing samples: {len(X_test_c)}")
    
    batch_size = 256
    train_dataset = TensorDataset(X_train_c, X_train_n, F_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = ForceConditionedKoopmanAE(input_dim=window_size, latent_dim=10, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_c, x_n, f in train_loader:
            x_c, x_n, f = x_c.to(device), x_n.to(device), f.to(device)
            optimizer.zero_grad()
            x_pred, x_next_pred, z, z_next = model(x_c, f)
            loss = criterion(x_pred, x_c) + criterion(x_next_pred, x_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss (Log-Space): {total_loss/len(train_loader):.6e}")
            
    print("Training complete.")
    
    model.eval()
    with torch.no_grad():
        x_c_test, x_n_test, f_test = X_test_c.to(device), X_test_n.to(device), F_test.to(device)
        _, x_next_pred, _, _ = model(x_c_test, f_test)
        test_loss = criterion(x_next_pred, x_n_test).item()
        print(f"Test Set Prediction MSE (Log-Space) on unseen forces: {test_loss:.6e}")
        
    save_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "koopman_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def prepare_rollout_windows(series_dict, window, horizon, f_norm):
    """series_dict: f -> 1D normalised series. Returns x0 (N, window), future windows
    (N, horizon, window) and forces (N, 1) for multi-step training."""
    X0, XF, F = [], [], []
    for f, s in series_dict.items():
        n = len(s) - window - horizon
        if n <= 0:
            continue
        idx = np.arange(window)[None, :] + np.arange(n)[:, None]
        W = s[idx]                                                   # (n, window)
        fut = np.stack([s[idx + k] for k in range(1, horizon + 1)], axis=1)
        X0.append(W)
        XF.append(fut)
        F.append(np.full((n, 1), f_norm(f)))
    return (torch.tensor(np.concatenate(X0), dtype=torch.float32),
            torch.tensor(np.concatenate(XF), dtype=torch.float32),
            torch.tensor(np.concatenate(F), dtype=torch.float32))


def train_rollout(series_dict, f_norm, window=20, horizon=40, latent_dim=10, hidden_dim=64,
                  epochs=30, batch_size=512, lr=1e-3, w_latent=0.1, seed=0, verbose=True):
    """
    Force-conditioned Koopman AE trained for FREE-RUNNING prediction: the loss compares
    decoder(K(f)^k z_0) with the true window k steps ahead for k = 1..horizon, plus a
    latent-consistency term ||K^k z_0 - enc(x_k)||. Unlike train_model(), the model never
    sees ground truth after the initial window inside the loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X0, XF, F = prepare_rollout_windows(series_dict, window, horizon, f_norm)
    loader = DataLoader(TensorDataset(X0, XF, F), batch_size=batch_size, shuffle=True)
    model = ForceConditionedKoopmanAE(input_dim=window, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mse = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x0, xf, f in loader:
            x0, xf, f = x0.to(device), xf.to(device), f.to(device)
            K = model.koopman_matrix(f)
            z = model.encoder(x0)
            loss = mse(model.decoder(z), x0)
            pred_loss, lat_loss = 0.0, 0.0
            for k in range(horizon):
                z = torch.bmm(K, z.unsqueeze(2)).squeeze(2)
                pred_loss = pred_loss + mse(model.decoder(z), xf[:, k])
                lat_loss = lat_loss + mse(z, model.encoder(xf[:, k]))
            loss = loss + pred_loss / horizon + w_latent * lat_loss / horizon
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"   epoch {ep + 1}/{epochs}  loss {tot / len(loader):.4e}")
    model.eval()
    return model


if __name__ == "__main__":
    train_model()
