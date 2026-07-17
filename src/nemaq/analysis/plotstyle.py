"""Shared publication figure style + dataset display metadata.

Every figure module imports from here so the paper's figures are stylistically
uniform (serif, 300 dpi, consistent class palettes across EDA / results / XAI).
"""
import matplotlib.pyplot as plt

PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
}

CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# Branch palette (fusion / attribution figures)
BRANCH_COLORS = {"q": "#9467bd", "geo": "#2ca02c", "bypass": "#ff7f0e"}
BRANCH_LABELS = {"q": "Quantum (PQC)", "geo": "Hyperbolic (geo)",
                 "bypass": "Classical bypass (trunk)"}

CLASS_NAMES = {
    "cora": ["Case Based", "Genetic Algorithms", "Neural Networks",
             "Probabilistic Meth.", "Reinforcement L.", "Rule Learning",
             "Theory"],
    "citeseer": ["Agents", "AI", "DB", "IR", "ML", "HCI"],
    "pubmed": ["Diabetes Exp.", "Diabetes T1", "Diabetes T2"],
    "disease": ["Not infected", "Infected"],
}

MODEL_LABELS = {
    "gcn": "GCN",
    "gat": "GAT",
    "mlp": "MLP",
    "hgcn": "HGCN",
    "nemaq:pqc:hyperbolic:residual": "NEMA-Q (full)",
    "nemaq:pqc:euclidean:residual": "NEMA-E (Euclidean)",
    "nemaq:frozen_random:hyperbolic:residual": "NEMA-R (frozen PQC)",
    "nemaq:surrogate:hyperbolic:residual": "NEMA-C (surrogate)",
    "nemaq:off:off:residual": "Trunk-only",
    "nemaq:pqc:hyperbolic:softmax": "NEMA-Q (softmax fusion)",
}


def class_names_for(dataset: str, num_classes: int) -> list[str]:
    names = CLASS_NAMES.get(dataset.lower())
    if names and len(names) == num_classes:
        return names
    return [f"C{c}" for c in range(num_classes)]


def apply_style():
    plt.rcParams.update(PUB_RC)
