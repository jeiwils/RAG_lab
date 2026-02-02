"""Shared helpers for supervised learning workflows."""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Mapping, Sequence

DEFAULT_GRID_CONFIG: Dict[str, Sequence[Any]] = {
    "hard_negatives": [6, 8],
    "random_negatives": [1],
    "pos_weight": [None, 2.0],
    "lr": [2e-5, 3e-5],
    "num_epochs": [8],
    "lora_r": [8],
    "lora_alpha": [16],
    "lora_dropout": [0.05],
    "batch_size": [16],
}

_GRID_KEYS = (
    "hard_negatives",
    "random_negatives",
    "pos_weight",
    "lr",
    "num_epochs",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "batch_size",
)


def grid_search(
    train_fn: Callable[..., Dict[str, Any]],
    *,
    output_dir: str,
    grid_config: Mapping[str, Sequence[Any]] | None = None,
    train_kwargs: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run a simple grid search over common hyperparameters."""
    resolved_grid = dict(DEFAULT_GRID_CONFIG)
    if grid_config:
        resolved_grid.update(grid_config)

    train_kwargs = dict(train_kwargs or {})
    train_kwargs.pop("output_dir", None)

    best: Dict[str, Any] = {"dev_f1": -1.0}
    run_idx = 0
    grid_values = [resolved_grid[key] for key in _GRID_KEYS]

    for values in itertools.product(*grid_values):
        params = dict(zip(_GRID_KEYS, values))
        run_idx += 1
        run_dir = (
            f"{output_dir}/grid"
            f"/run{run_idx}"
            f"_hn{params['hard_negatives']}"
            f"_rn{params['random_negatives']}"
            f"_pw{params['pos_weight']}"
            f"_lr{params['lr']}"
            f"_ep{params['num_epochs']}"
            f"_r{params['lora_r']}"
            f"_a{params['lora_alpha']}"
            f"_d{params['lora_dropout']}"
            f"_bs{params['batch_size']}"
        )
        metrics = train_fn(
            hard_negatives=params["hard_negatives"],
            random_negatives=params["random_negatives"],
            pos_weight=params["pos_weight"],
            lr=params["lr"],
            num_epochs=params["num_epochs"],
            batch_size=params["batch_size"],
            lora_r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            output_dir=run_dir,
            **train_kwargs,
        )
        metrics["run_dir"] = run_dir
        metrics["run_idx"] = run_idx
        if metrics.get("dev_f1", -1.0) > best.get("dev_f1", -1.0):
            best = metrics
        print(
            f"[grid {run_idx}] dev_f1={metrics['dev_f1']:.4f} "
            f"tau={metrics['tau']:.3f} dir={run_dir}"
        )

    print(
        f"[grid best] dev_f1={best['dev_f1']:.4f} "
        f"tau={best['tau']:.3f} dir={best.get('run_dir')}"
    )
    return best


__all__ = ["DEFAULT_GRID_CONFIG", "grid_search"]
