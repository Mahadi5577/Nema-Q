"""Poincare-ball hyperbolic graph encoder (geoopt).

HGCN-style tangent-space message passing:
  1. exp-map features to the ball at the origin,
  2. per-layer: Mobius linear transform, log-map to tangent space,
     neighborhood mean-aggregate, exp-map back, pointwise nonlinearity
     applied in tangent space,
  3. final log-map so downstream fusion operates in a shared tangent space.

Curvature c is learnable (softplus-positive) unless frozen by config —
learned curvature per dataset is itself reportable evidence for H1.
"""
import geoopt
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import add_self_loops, degree


class HypLinear(nn.Module):
    def __init__(self, manifold: geoopt.PoincareBall, in_dim: int, out_dim: int):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        mv = self.manifold.mobius_matvec(self.weight, x)
        b = self.manifold.expmap0(self.bias)
        return self.manifold.mobius_add(mv, b)


class HypGraphConv(nn.Module):
    """Aggregate in tangent space at the origin (mean over neighbors, incl. self)."""

    def __init__(self, manifold: geoopt.PoincareBall, in_dim: int, out_dim: int,
                 dropout: float = 0.5, act: bool = True):
        super().__init__()
        self.manifold = manifold
        self.lin = HypLinear(manifold, in_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index):
        x = self.lin(x)
        t = self.manifold.logmap0(x)
        ei, _ = add_self_loops(edge_index, num_nodes=t.size(0))
        row, col = ei
        deg = degree(row, t.size(0), dtype=t.dtype).clamp(min=1)
        agg = torch.zeros_like(t).index_add_(0, row, t[col]) / deg.unsqueeze(-1)
        if self.act:
            agg = F.relu(agg)
        agg = F.dropout(agg, p=self.dropout, training=self.training)
        return self.manifold.projx(self.manifold.expmap0(agg))


MAX_TANGENT_NORM = 2.5  # keep exp-mapped points away from the ball boundary


def clip_tangent(v: torch.Tensor, max_norm: float = MAX_TANGENT_NORM) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return v * (max_norm / norm).clamp(max=1.0)


class HyperbolicEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float = 0.5, c: float = 1.0, learnable_c: bool = True):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c, learnable=learnable_c)
        # Euclidean compression before the exp-map: mapping raw high-dim
        # features (e.g. 1433-dim Cora bags-of-words) straight onto the ball
        # saturates points at the boundary where gradients vanish.
        self.feat = nn.Linear(in_dim, hidden)
        self.conv1 = HypGraphConv(self.manifold, hidden, hidden, dropout)
        self.conv2 = HypGraphConv(self.manifold, hidden, out_dim, dropout, act=False)

    def forward(self, x, edge_index):
        t = clip_tangent(self.feat(x))
        h = self.manifold.projx(self.manifold.expmap0(t))
        h = self.conv1(h, edge_index)
        h = self.conv2(h, edge_index)
        return self.manifold.logmap0(h)  # tangent-space output for fusion

    @property
    def curvature(self) -> float:
        return float(self.manifold.c.detach())


class HGCN(nn.Module):
    """Standalone hyperbolic GNN baseline (ICEQT'26 referee B: 'add ...
    hyperbolic GNN baselines under the same split')."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float = 0.5, c: float = 1.0, learnable_c: bool = True):
        super().__init__()
        self.encoder = HyperbolicEncoder(in_dim, hidden, hidden, dropout,
                                         c=c, learnable_c=learnable_c)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        return self.head(self.encoder(x, edge_index))

    @property
    def curvature(self) -> float:
        return self.encoder.curvature
