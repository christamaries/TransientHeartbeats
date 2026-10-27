"""
Validate a trained PulseCountCNN on the held-out test set saved by train.py,
and plot the true vs. predicted pulse-count distribution.

Usage:
    conda run -n pygad3 python validate.py
"""
import argparse
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import PulseCountCNN

# Colors from the project's validated categorical palette (dataviz skill):
# slot 1 = blue (true), slot 2 = green (predicted) -- fixed order, not cycled.
COLOR_TRUE = '#2a78d6'
COLOR_PRED = '#008300'
COLOR_INK = '#0b0b0b'
COLOR_SUBTLE = '#52514e'
COLOR_MUTED = '#898781'
COLOR_GRID = '#e1e0d9'
COLOR_BASELINE = '#c3c2b7'
COLOR_SURFACE = '#fcfcfb'


def plot_distribution(y_true, y_pred, out_path):
    max_count = int(max(y_true.max(), y_pred.max()))
    x = np.arange(1, max_count + 1)
    true_counts = np.array([(y_true == k).sum() for k in x])
    pred_counts = np.array([(y_pred == k).sum() for k in x])

    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    ax.bar(x - width / 2 - 0.02, true_counts, width=width, color=COLOR_TRUE,
           label='True', zorder=3)
    ax.bar(x + width / 2 + 0.02, pred_counts, width=width, color=COLOR_PRED,
           label='Predicted', zorder=3)

    ax.set_xticks(x)
    ax.set_xlabel('Number of parent pulses', color=COLOR_SUBTLE)
    ax.set_ylabel('Number of light curves', color=COLOR_SUBTLE)
    ax.set_title('True vs. predicted pulse-count distribution (test set)',
                 color=COLOR_INK, fontsize=13, loc='left')

    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        if name in ('top', 'right'):
            spine.set_visible(False)
        else:
            spine.set_color(COLOR_BASELINE)
    ax.tick_params(colors=COLOR_MUTED)

    ax.legend(frameon=False, labelcolor=COLOR_SUBTLE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved distribution plot to {out_path}")


def plot_error_histogram(y_true, y_pred, out_path):
    """Histogram of (predicted - true): how far off, and in which direction."""
    errors = y_pred - y_true
    lo, hi = int(errors.min()), int(errors.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1)  # integer-centered bins

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    ax.hist(errors, bins=bins, color=COLOR_TRUE, zorder=3, rwidth=0.85)
    ax.axvline(0, color=COLOR_MUTED, linewidth=1.5, linestyle='--', zorder=4)

    ax.set_xticks(np.arange(lo, hi + 1))
    ax.set_xlabel('Predicted - True (parent pulses)', color=COLOR_SUBTLE)
    ax.set_ylabel('Number of light curves', color=COLOR_SUBTLE)
    ax.set_title('Prediction error distribution (test set)',
                 color=COLOR_INK, fontsize=13, loc='left')

    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        if name in ('top', 'right'):
            spine.set_visible(False)
        else:
            spine.set_color(COLOR_BASELINE)
    ax.tick_params(colors=COLOR_MUTED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved error histogram to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='./runs',
                         help='Directory containing best_model.pt and test_split.npz from train.py')
    parser.add_argument('--out', default=None, help='Output path for the histogram PNG')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run_dir, 'best_model.pt')
    split_path = os.path.join(args.run_dir, 'test_split.npz')
    if not os.path.exists(ckpt_path) or not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing {ckpt_path} or {split_path} -- run train.py first.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PulseCountCNN().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    data = np.load(split_path)
    X_test, y_test = data['X_test'], data['y_test']

    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    preds_round = np.clip(np.round(preds), 1, None).astype(int)

    mae = np.mean(np.abs(preds_round - y_test))
    acc = np.mean(preds_round == y_test)
    within1 = np.mean(np.abs(preds_round - y_test) <= 1)
    print(f"Test set size: {len(y_test)}")
    print(f"MAE: {mae:.3f}")
    print(f"Exact-match accuracy: {acc:.3f}")
    print(f"Within +/-1 accuracy: {within1:.3f}")

    print("Per-class test accuracy:")
    for k in sorted(np.unique(y_test)):
        mask = y_test == k
        n_k = mask.sum()
        acc_k = (preds_round[mask] == k).mean() if n_k > 0 else float('nan')
        print(f"  n_parent_pulses={k}: n={n_k:4d}  exact-match={acc_k:.3f}")

    out_path = args.out or os.path.join(args.run_dir, 'pulse_count_distribution.png')
    plot_distribution(y_test, preds_round, out_path)

    error_out_path = os.path.join(os.path.dirname(out_path), 'prediction_error_histogram.png')
    plot_error_histogram(y_test, preds_round, error_out_path)


if __name__ == '__main__':
    main()
