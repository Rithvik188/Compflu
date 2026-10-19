"""
Force-Conditioned Deep Koopman Autoencoder.
Generates a linear Koopman operator in latent space dynamically based on the active force `f`.
"""
import torch
import torch.nn as nn

class ForceConditionedKoopmanAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super(ForceConditionedKoopmanAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Maps raw trajectory window to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Sub-network to generate the Koopman matrix from active force
        self.koopman_generator = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * latent_dim)
        )
        
        # Maps latent space back to trajectory window
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, f):
        """
        x: (batch, input_dim) - current state window
        f: (batch, 1) - active force
        """
        # Encode
        z = self.encoder(x)
        
        # Generate Koopman operator K for each sample in batch based on its force f
        # Output shape: (batch, latent_dim, latent_dim)
        K_flat = self.koopman_generator(f)
        K = K_flat.view(-1, self.latent_dim, self.latent_dim)
        
        # Advance in latent space: z_next = K * z
        z_next = torch.bmm(K, z.unsqueeze(2)).squeeze(2)
        
        # Decode
        x_pred = self.decoder(z)
        x_next_pred = self.decoder(z_next)
        
        return x_pred, x_next_pred, z, z_next

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Force-Conditioned Deep Koopman Autoencoder targeting device: {device}")
    
    model = ForceConditionedKoopmanAE(input_dim=50, latent_dim=10).to(device)
    dummy_input = torch.randn(32, 50).to(device)
    dummy_force = torch.randn(32, 1).to(device)
    
    x_pred, x_next_pred, z, z_next = model(dummy_input, dummy_force)
    print(f"Forward pass successful. Next state prediction shape: {x_next_pred.shape}")
