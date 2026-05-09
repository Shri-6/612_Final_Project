"""
Generate the "Attention vs Rolling Correlation" heatmap figure for the slide deck.

Reads from the existing artifacts produced by:
    scripts/05_extract_attention.py

Outputs:
    artifacts/figures/attention_vs_correlation.png   (high-res, slide-ready)

The figure is two side-by-side heatmaps that match the visual style of the
reference slide (navy text, red-blue diverging for correlation, blues for
attention, ticker labels on both axes).

Run from project root:
    python scripts/make_attention_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "artifacts"
OUT_DIR = ART / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "attention_vs_correlation.png"

# Aggregation: which day to show, or "mean" to average across all test days.
# "mean" gives the cleanest visual for a slide; pick a specific day index
# (e.g. AGG = 100) if you want a snapshot.
AGG = "mean"

# Color palette tuned to match the slide template (navy primary)
NAVY = "#0B3D91"
TEXT_DARK = "#1A1A1A"


# -----------------------------------------------------------------------------
def load_arrays():
    """Load attention + rolling-correlation arrays + ticker labels."""
    a_npz = np.load(ART / "attention" / "test_attention.npz", allow_pickle=False)
    r_npz = np.load(ART / "attention" / "test_corr_rolling.npz", allow_pickle=False)

    # Pick the (T, N, N) array out of each file (key naming has varied)
    A = next((a_npz[k] for k in a_npz.files if a_npz[k].ndim == 3), None)
    R = next((r_npz[k] for k in r_npz.files if r_npz[k].ndim == 3), None)
    if A is None or R is None:
        raise RuntimeError("Could not find (T, N, N) arrays in attention files")

    # Ticker labels from meta.json
    meta = json.load(open(ART / "data" / "meta.json"))
    tickers = list(meta["tickers"])

    # Sanity check
    assert A.shape[1] == R.shape[1] == len(tickers), \
        f"Asset count mismatch: attn={A.shape[1]}, corr={R.shape[1]}, tickers={len(tickers)}"

    return A, R, tickers


def aggregate(arr3d, mode):
    """Reduce a (T, N, N) array to (N, N) for plotting."""
    if mode == "mean":
        # nanmean to be robust to any NaNs from the 60-day rolling window warmup
        return np.nanmean(arr3d, axis=0)
    if isinstance(mode, int):
        return arr3d[mode]
    raise ValueError(f"Unknown agg mode: {mode}")


# -----------------------------------------------------------------------------
def plot(A2d, R2d, tickers, out_path):
    """Render two side-by-side heatmaps with consistent styling."""
    n = len(tickers)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"wspace": 0.35})

    # --- LEFT: Rolling Pearson correlation (RdBu_r diverging, [-1, 1]) -------
    ax = axes[0]
    im = ax.imshow(R2d, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ax.set_title("Rolling Pearson correlation", fontsize=14, fontweight="bold",
                 color=NAVY, pad=12)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=90, fontsize=9, color=TEXT_DARK)
    ax.set_yticklabels(tickers, fontsize=9, color=TEXT_DARK)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_DARK)
    cbar.outline.set_visible(False)

    # --- RIGHT: Cross-asset attention (Blues, raw range) ---------------------
    ax = axes[1]
    im = ax.imshow(A2d, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_title("Cross-asset attention (layer 3, head-mean)", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=90, fontsize=9, color=TEXT_DARK)
    ax.set_yticklabels(tickers, fontsize=9, color=TEXT_DARK)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_DARK)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote: {out_path}")


# -----------------------------------------------------------------------------
def main():
    A3d, R3d, tickers = load_arrays()
    print(f"Loaded:  attention {A3d.shape}, correlation {R3d.shape}, "
          f"{len(tickers)} tickers")

    A2d = aggregate(A3d, AGG)
    R2d = aggregate(R3d, AGG)

    plot(A2d, R2d, tickers, OUT_PATH)
    print()
    print(f"Done. Drag {OUT_PATH.relative_to(ROOT)} into your slide.")


if __name__ == "__main__":
    main()
