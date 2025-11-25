import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    """
    Graph Attention Encoder (Simplified 1-Layer Version).
    Directly maps input features to latent space using attention.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads=2, dropout=0.2):
        super(GATEncoder, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        x = self.dropout(x)
        x = nn.functional.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        
        return x

class FeatureDecoder(nn.Module):
    """
    Reconstructs the original node features (RNN embeddings) from the latent Z.
    """
    def __init__(self, latent_channels: int, hidden_channels: int, out_channels: int):
        super(FeatureDecoder, self).__init__()
        # Simple MLP decoder
        self.lin1 = nn.Linear(latent_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, z):
        x = F.relu(self.lin1(z))
        x = self.lin2(x)
        return x

class GATEModel(nn.Module):
    """
    End-to-End Graph Attention Autoencoder.
    """
    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int):
        super(GATEModel, self).__init__()
        # Removed hidden_channels argument as we only have 1 layer
        self.encoder = GATEncoder(in_channels, hidden_channels, latent_channels)
        self.feature_decoder = FeatureDecoder(latent_channels, hidden_channels, in_channels)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        z = self.encoder(x, edge_index)
        return z

    def recon_loss(self, z: torch.tensor, x: torch.tensor, edge_index: torch.tensor, lambda_feat=1, lambda_struct=0.1):
        """
        Dual Reconstruction Loss.
        """
        # 1. Feature Reconstruction Loss - Sum Squared Error
        x_hat = self.feature_decoder(z)
        loss_feat = F.mse_loss(x_hat, x)

        # 2. Structure Reconstruction Loss - Only backprop through connected edges, not unconnected ones
        adj_hat = torch.sigmoid(torch.matmul(z, z.t()))
        loss_struct = adj_hat[edge_index[0], edge_index[1]].mean()

        return (lambda_feat * loss_feat) + (lambda_struct * loss_struct), loss_feat.item(), loss_struct.item()