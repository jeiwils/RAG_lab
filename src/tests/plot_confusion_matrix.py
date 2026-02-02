"""Plot a confusion matrix from best_metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path



def _load_confusion(metrics_path: Path) -> dict[str, int]:
    with open(metrics_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    confusion = data.get("dev_confusion")
    if not isinstance(confusion, dict):
        raise ValueError("dev_confusion not found in metrics file")
    return {
        "tn": int(confusion.get("tn", 0)),
        "fp": int(confusion.get("fp", 0)),
        "fn": int(confusion.get("fn", 0)),
        "tp": int(confusion.get("tp", 0)),
    }


def plot_confusion(metrics_path: Path, output_path: Path | None) -> Path:
    import matplotlib.pyplot as plt

    counts = _load_confusion(metrics_path)
    matrix = [
        [counts["tn"], counts["fp"]],
        [counts["fn"], counts["tp"]],
    ]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if output_path is None:
        output_path = metrics_path.with_suffix(".confusion.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metrics",
        type=Path,
        help="Path to best_metrics.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path for PNG",
    )
    args = parser.parse_args()

    output_path = plot_confusion(args.metrics, args.out)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
