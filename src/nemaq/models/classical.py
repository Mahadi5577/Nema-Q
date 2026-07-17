"""Classical baselines and the classical bypass branch.

These must reproduce published Cora/Citeseer/Pubmed numbers (Gate 2)
before any hybrid experiment counts.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_dim, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class MLP(nn.Module):
    """Graph-blind control: any model beating this proves structure matters."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index=None):
        return self.net(x)


class GCNEncoder(nn.Module):
    """Bypass branch: GCN body without classifier head, used inside NEMA-Q."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
