"""
Core implementation of Hankel Dynamic Mode Decomposition (HDMD).
Designed for extracting Koopman eigenvalues from 1D time-series data.
"""
import numpy as np
import torch

class KoopmanHDMD:
    def __init__(self, delay_window: int, rank: int, use_gpu: bool = True):
        self.delay_window = m = delay_window
        self.rank = rank
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"HDMD Initialized on device: {self.device}")

    def build_hankel_matrix(self, time_series: np.ndarray) -> torch.Tensor:
        """Constructs the Hankel matrix from a 1D array."""
        N = len(time_series)
        columns = N - self.delay_window + 1
        H = np.array([time_series[i:i+columns] for i in range(self.delay_window)])
        return torch.tensor(H, dtype=torch.float32, device=self.device)

    def fit(self, time_series: np.ndarray):
        """Performs SVD and extracts the Koopman operator and exact decay rates."""
        H = self.build_hankel_matrix(time_series)
        
        X = H[:, :-1]
        Y = H[:, 1:]
        
        # Compute SVD on the GPU (or CPU fallback)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        
        # Truncate
        U_r = U[:, :self.rank]
        S_r = torch.diag(S[:self.rank])
        V_r = Vh[:self.rank, :].conj().T
        
        # Compute Koopman operator (A_tilde)
        S_r_inv = torch.linalg.inv(S_r)
        A_tilde = U_r.conj().T @ Y @ V_r @ S_r_inv
        
        # Extract Eigenvalues (representing decay rates / relaxation times)
        eigenvalues, _ = torch.linalg.eig(A_tilde)
        
        self.eigenvalues = eigenvalues.cpu().numpy()
        self.A_tilde = A_tilde.cpu().numpy()
        return self.eigenvalues

if __name__ == "__main__":
    # Dummy Test
    dummy_data = np.exp(-0.01 * np.linspace(0, 100, 1000))
    model = KoopmanHDMD(delay_window=50, rank=10)
    eigs = model.fit(dummy_data)
    print(f"Extracted {len(eigs)} eigenvalues successfully.")
