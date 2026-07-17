"""Dataset loading with controlled split protocols.

Split modes:
  - "public":   the standard Planetoid split (comparability with literature).
  - "low_label": k labels per class, seeded — the low-label regime for H2.
  - "ratio":    seeded stratified ratio split (default 30/10/60, HGCN
                protocol) — for datasets without a public split (Disease).
Both return masks on the Data object; the split seed is independent of the
model seed so the same splits are reused across models (paired statistics).
"""
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
AMAZON = {"photo": "Photo", "computers": "Computers"}
WEBKB = {"cornell": "Cornell", "texas": "Texas", "wisconsin": "Wisconsin"}

# Disease (Chami et al. 2019, HGCN repo): synthetic SIR propagation tree,
# delta ~= 0 — the positive control for H1. No public split; use split
# mode "ratio" (HGCN protocol: 30/10/60).
_DISEASE_BASE = ("https://raw.githubusercontent.com/HazyResearch/hgcn/"
                 "master/data/disease_nc/")
_DISEASE_FILES = ("disease_nc.edges.csv", "disease_nc.feats.npz",
                  "disease_nc.labels.npy")


class _InMemoryDataset(list):
    """Minimal ds wrapper so callers can keep using ds[0]."""

    def __init__(self, data: Data):
        super().__init__([data])
        self.num_features = data.num_features
        self.num_classes = int(data.y.max()) + 1


def _load_disease(root: str) -> _InMemoryDataset:
    import scipy.sparse as sp

    ddir = Path(root) / "disease_nc"
    ddir.mkdir(parents=True, exist_ok=True)
    for fname in _DISEASE_FILES:
        dest = ddir / fname
        if not dest.exists():
            urllib.request.urlretrieve(_DISEASE_BASE + fname, dest)

    edges = []
    with open(ddir / "disease_nc.edges.csv", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 2 or not parts[0].isdigit():
                continue  # header or malformed line
            edges.append((int(parts[0]), int(parts[1])))
    edge_index = to_undirected(torch.tensor(edges, dtype=torch.long).t())

    feats = sp.load_npz(ddir / "disease_nc.feats.npz").todense()
    x = torch.tensor(np.asarray(feats), dtype=torch.float)
    x = x / x.sum(dim=-1, keepdim=True).clamp(min=1)  # row-normalize
    y = torch.tensor(np.load(ddir / "disease_nc.labels.npy"),
                     dtype=torch.long).view(-1)
    return _InMemoryDataset(Data(x=x, edge_index=edge_index, y=y))


def load_dataset(name: str, root: str = "data"):
    name = name.lower()
    if name in PLANETOID:
        ds = Planetoid(root, PLANETOID[name], transform=NormalizeFeatures())
    elif name in AMAZON:
        ds = Amazon(root, AMAZON[name], transform=NormalizeFeatures())
    elif name in WEBKB:
        ds = WebKB(root, WEBKB[name], transform=NormalizeFeatures())
    elif name == "disease":
        ds = _load_disease(root)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds


def apply_split(data, mode: str = "public", labels_per_class: int = 5,
                split_seed: int = 0, val_size: int = 500, test_size: int = 1000,
                **kwargs):
    if mode == "public":
        if not hasattr(data, "train_mask") or data.train_mask is None:
            raise ValueError("Dataset has no public split; use mode='low_label'.")
        # WebKB ships multiple splits as [N, 10] masks — take column split_seed % 10
        if data.train_mask.dim() == 2:
            col = split_seed % data.train_mask.size(1)
            data.train_mask = data.train_mask[:, col]
            data.val_mask = data.val_mask[:, col]
            data.test_mask = data.test_mask[:, col]
        return data

    if mode == "ratio":
        # HGCN protocol for datasets without a public split (Disease):
        # seeded stratified 30/10/60 by default.
        g = torch.Generator().manual_seed(split_seed)
        n, y = data.num_nodes, data.y
        train_ratio = kwargs.get("train_ratio", 0.30)
        val_ratio = kwargs.get("val_ratio", 0.10)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        for c in y.unique():  # stratified: preserve class balance per split
            idx = (y == c).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(len(idx), generator=g)]
            n_tr = max(1, int(round(train_ratio * len(perm))))
            n_va = max(1, int(round(val_ratio * len(perm))))
            train_mask[perm[:n_tr]] = True
            val_mask[perm[n_tr:n_tr + n_va]] = True
            test_mask[perm[n_tr + n_va:]] = True
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
        return data

    if mode == "low_label":
        g = torch.Generator().manual_seed(split_seed)
        n, y = data.num_nodes, data.y
        train_mask = torch.zeros(n, dtype=torch.bool)
        for c in y.unique():
            idx = (y == c).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(len(idx), generator=g)]
            train_mask[perm[:labels_per_class]] = True
        rest = (~train_mask).nonzero(as_tuple=True)[0]
        rest = rest[torch.randperm(len(rest), generator=g)]
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        val_mask[rest[:val_size]] = True
        test_mask[rest[val_size:val_size + test_size]] = True
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
        return data

    raise ValueError(f"Unknown split mode: {mode}")
