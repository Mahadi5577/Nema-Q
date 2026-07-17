"""Single-run trainer: full-batch node classification with early stopping,
gradient telemetry always on (instrumentation-first), manifest on exit.

Note: geoopt manifold parameters (learnable curvature) train correctly with
a Riemannian optimizer; we use geoopt.optim.RiemannianAdam, which reduces to
Adam for Euclidean parameters — one optimizer for every model variant.
"""
import copy
from pathlib import Path

import geoopt
import torch
import torch.nn.functional as F

from ..models import build_model
from ..telemetry.gradients import GradientTelemetry
from ..utils.manifest import write_manifest
from ..utils.seed import set_seed


def accuracy(logits, y, mask) -> float:
    return float((logits[mask].argmax(dim=-1) == y[mask]).float().mean())


def f1_per_class(logits, y, mask) -> list[float]:
    """Per-class F1 — min over classes is the minority-class-collapse
    detector (ICEQT'26 referee B flagged an unnoticed per-class F1 collapse)."""
    pred, yt = logits[mask].argmax(dim=-1), y[mask]
    scores = []
    for c in range(int(y.max()) + 1):
        tp = int(((pred == c) & (yt == c)).sum())
        fp = int(((pred == c) & (yt != c)).sum())
        fn = int(((pred != c) & (yt == c)).sum())
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return scores


def param_counts(model) -> dict[str, int]:
    counts = {"total": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if hasattr(model, "branches"):
        for name, b in model.branches.items():
            counts[f"branch_{name}"] = sum(p.numel() for p in b.parameters()
                                           if p.requires_grad)
    return counts


def train_run(cfg: dict, data, seed: int, run_dir: str | Path,
              return_model: bool = False):
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data = data.to(device)

    model = build_model(
        cfg["model"]["name"], data.num_features,
        int(data.y.max()) + 1, cfg["model"],
    ).to(device)

    # No weight decay on fusion gates (decay drags them back toward their
    # init and fights gate learning) or manifold curvature.
    decay, no_decay = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if ("fusion.gate" in pname or "manifold" in pname)
         else decay).append(p)
    opt = geoopt.optim.RiemannianAdam(
        [{"params": decay, "weight_decay": cfg["train"].get("weight_decay", 5e-4)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["train"].get("lr", 0.01),
    )
    telemetry = GradientTelemetry(model)
    aux_weight = cfg["train"].get("aux_weight", 0.3)
    aux_anneal = cfg["train"].get("aux_anneal", True)

    best_val, best_state, patience_left = -1.0, None, cfg["train"].get("patience", 100)
    max_epochs = cfg["train"].get("epochs", 500)

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        # deep supervision: every branch trains against the labels directly.
        # Annealed to zero so it prevents early gate-collapse starvation but
        # stops fighting the fused objective at convergence.
        w = aux_weight * (1 - epoch / max_epochs) if aux_anneal else aux_weight
        if w > 0 and getattr(model, "last_aux_logits", None):
            aux = sum(
                F.cross_entropy(a[data.train_mask], data.y[data.train_mask])
                for a in model.last_aux_logits.values()
            ) / len(model.last_aux_logits)
            loss = loss + w * aux
        loss.backward()
        telemetry.record(epoch)
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg["train"].get("patience", 100)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    f1s = f1_per_class(logits, data.y, data.test_mask)
    metrics = {
        "val_acc": best_val,
        "test_acc": accuracy(logits, data.y, data.test_mask),
        "test_macro_f1": sum(f1s) / len(f1s),
        "test_min_class_f1": min(f1s),   # 0.0 here = minority-class collapse
        "epochs_run": epoch + 1,
        "param_counts": param_counts(model),
    }
    if hasattr(model, "branches") and "geo" in model.branches \
            and hasattr(model.branches["geo"], "curvature"):
        metrics["learned_curvature"] = model.branches["geo"].curvature

    write_manifest(run_dir, cfg, seed, metrics, telemetry.summary())
    if return_model:
        return metrics, model, data
    return metrics
