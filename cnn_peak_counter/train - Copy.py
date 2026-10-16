"""
Train a 1D CNN to predict the number of parent pulses (n_parent_pulses) in a
simulated GRB light curve.

Usage:
    conda run -n pygad3 python train.py
    conda run -n pygad3 python train.py --run-dir ../simulated_LCs/2026-07-21_09-00-00
"""
import argparse
import os
import sys

# Torch's bundled OpenMP runtime (libiomp5md.dll) collides with the one numpy
# already loads on this Windows/conda setup; harmless here since we're single-
# process CPU training, not relying on strict OpenMP thread affinity.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Line-buffer stdout so per-epoch progress appears live in a redirected/
# background log instead of only flushing at process exit.
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from dataset import load_run, find_latest_run, cap_per_class
from model import PulseCountCNN


def _can_stratify(y):
    # sklearn's stratify requires every class to have >= 2 members
    _, counts = np.unique(y, return_counts=True)
    return counts.min() >= 2


def make_train_loader(X_, y_, batch_size, balanced):
    """
    balanced=True draws each minibatch with inverse-class-frequency
    weighting (via WeightedRandomSampler with replacement), so rare
    pulse-count classes appear roughly as often as common ones during
    training instead of being swamped by the dominant 1-2-pulse classes.
    See REPORT.md Sec. 7.5/7.7.
    """
    ds = TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_.astype(np.float32)))
    if not balanced:
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    y_int = y_.astype(int)
    class_counts = np.bincount(y_int, minlength=y_int.max() + 1)
    with np.errstate(divide='ignore'):
        class_weight = np.where(class_counts > 0, 1.0 / np.maximum(class_counts, 1), 0.0)
    sample_weights = class_weight[y_int]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(y_int),
        replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


class LiveTrainingPlot:
    """
    Opens a small on-screen window showing train/val loss and val MAE/accuracy
    updating live, one point per epoch. Opt-in via --live-plot, since it
    needs an actual display (a GUI backend + an interactive terminal you're
    watching) -- it's silently unavailable, not an error, in a headless or
    background-redirected run.
    """
    COLOR_TRAIN = '#2a78d6'
    COLOR_VAL = '#e34948'
    COLOR_MAE = '#008300'
    COLOR_ACC = '#eda100'
    COLOR_INK = '#0b0b0b'
    COLOR_SUBTLE = '#52514e'
    COLOR_GRID = '#e1e0d9'
    COLOR_SURFACE = '#fcfcfb'

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, (self.ax_loss, self.ax_metric) = plt.subplots(
            1, 2, figsize=(10, 4.2), facecolor=self.COLOR_SURFACE)
        try:
            self.fig.canvas.manager.set_window_title('Training progress')
        except Exception:
            pass
        self.epochs, self.train_loss, self.val_loss, self.val_mae, self.val_acc = [], [], [], [], []
        self.fig.show()
        self.plt.pause(0.01)

    def _style(self, ax, title, ylabel):
        ax.set_facecolor(self.COLOR_SURFACE)
        ax.set_xlabel('Epoch', color=self.COLOR_SUBTLE)
        ax.set_ylabel(ylabel, color=self.COLOR_SUBTLE)
        ax.set_title(title, color=self.COLOR_INK, fontsize=11, loc='left')
        ax.yaxis.grid(True, color=self.COLOR_GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for name, spine in ax.spines.items():
            spine.set_visible(name not in ('bottom', 'left'))
        ax.tick_params(colors=self.COLOR_SUBTLE)
        ax.legend(frameon=False, labelcolor=self.COLOR_SUBTLE)

    def update(self, epoch, train_loss, val_loss, val_mae, val_acc):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_mae.append(val_mae)
        self.val_acc.append(val_acc)

        self.ax_loss.clear()
        self.ax_loss.plot(self.epochs, self.train_loss, color=self.COLOR_TRAIN, label='Train loss')
        self.ax_loss.plot(self.epochs, self.val_loss, color=self.COLOR_VAL, label='Val loss')
        self._style(self.ax_loss, 'Loss', 'Loss')

        self.ax_metric.clear()
        self.ax_metric.plot(self.epochs, self.val_mae, color=self.COLOR_MAE, label='Val MAE')
        self.ax_metric.plot(self.epochs, self.val_acc, color=self.COLOR_ACC, label='Val exact-match acc')
        self._style(self.ax_metric, 'Validation metrics', 'Value')

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def keep_open(self):
        print("Close the training-progress window to continue...")
        self.plt.ioff()
        self.plt.show()


def per_class_report(y_true, y_pred_round, title='Per-class test accuracy'):
    print(title + ':')
    for k in sorted(np.unique(y_true)):
        mask = y_true == k
        n_k = mask.sum()
        acc_k = (y_pred_round[mask] == k).mean() if n_k > 0 else float('nan')
        print(f"  n_parent_pulses={k}: n={n_k:4d}  exact-match={acc_k:.3f}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_abs_err = 0.0
    total_correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * len(xb)
            pred_round = torch.clamp(torch.round(pred), min=1)
            total_abs_err += torch.abs(pred_round - yb).sum().item()
            total_correct += (pred_round == yb).sum().item()
            n += len(xb)
    return total_loss / n, total_abs_err / n, total_correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=None,
                         help='Path to a simulated_LCs/<timestamp> directory. '
                              'Defaults to the most recently generated run.')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--test-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='./runs')
    parser.add_argument('--sampler', choices=['uniform', 'balanced'], default='balanced',
                         help="'balanced' uses inverse-class-frequency weighted sampling "
                              "for training batches, to counter the class imbalance in "
                              "n_parent_pulses (see REPORT.md Sec. 7.5). 'uniform' "
                              "reproduces the original (pre-fix) behavior.")
    parser.add_argument('--loss', choices=['mse', 'poisson'], default='poisson',
                         help="'poisson' uses Poisson NLL loss, matching the count-data "
                              "nature of n_parent_pulses. 'mse' reproduces the original "
                              "(pre-fix) behavior.")
    parser.add_argument('--preset', choices=['baseline', 'poisson-only', 'fix', 'mse-weighted'],
                         default=None,
                         help="Shortcut that sets --sampler/--loss together, so the same "
                              "configurations can be run against any --run-dir "
                              "(e.g. the 3,000- or 10,000-LC dataset) without remembering "
                              "flag combinations. Overrides --sampler/--loss if given. "
                              "'baseline' = uniform sampler + MSE loss (original, pre-fix; "
                              "aka mse_unw). "
                              "'poisson-only' = uniform sampler + Poisson loss (aka "
                              "poisson_unw; isolates the loss change alone, no reweighting). "
                              "'fix' = balanced sampler + Poisson loss (aka poisson_weighted; "
                              "the full fix, §7.6). "
                              "'mse-weighted' = balanced sampler + MSE loss (aka mse_weighted; "
                              "completes the 2x2 sampler x loss factorial, §7.10).")
    parser.add_argument('--live-plot', action='store_true',
                         help='Open an on-screen window showing loss/MAE/accuracy updating '
                              'live, one point per epoch. Needs an interactive display; '
                              'silently skipped (with a warning) if none is available.')
    parser.add_argument('--cap-per-class', type=int, default=None,
                         help='Undersample the dataset so each n_parent_pulses value has '
                              'at most this many examples (keeping all available for classes '
                              'already below the cap), applied before the train/val/test '
                              'split. A different, complementary way to counter class '
                              'imbalance from --sampler balanced -- this shrinks the training '
                              'set itself rather than reweighting how it is sampled.')
    args = parser.parse_args()

    if args.preset is not None:
        args.sampler, args.loss = {
            'baseline':     ('uniform',  'mse'),
            'poisson-only': ('uniform',  'poisson'),
            'fix':          ('balanced', 'poisson'),
            'mse-weighted': ('balanced', 'mse'),
        }[args.preset]

    live_plot = None
    if args.live_plot:
        try:
            live_plot = LiveTrainingPlot()
        except Exception as e:
            print(f"--live-plot requested but could not open a display ({e}); continuing without it.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = args.run_dir or find_latest_run()
    print(f"Loading light curves from: {run_dir}")

    X, y = load_run(run_dir)
    print(f"Loaded {len(X)} light curves, length {X.shape[1]} bins.")
    values, counts = np.unique(y, return_counts=True)
    print("Label distribution (n_parent_pulses -> count):")
    for v, c in zip(values, counts):
        print(f"  {v}: {c}")

    if args.cap_per_class is not None:
        X, y = cap_per_class(X, y, args.cap_per_class, seed=args.seed)
        values, counts = np.unique(y, return_counts=True)
        print(f"After capping at {args.cap_per_class}/class: {len(X)} light curves.")
        print("Capped label distribution (n_parent_pulses -> count):")
        for v, c in zip(values, counts):
            print(f"  {v}: {c}")

    strat = y if _can_stratify(y) else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=args.val_frac + args.test_frac,
        random_state=args.seed, stratify=strat)

    strat_tmp = y_tmp if _can_stratify(y_tmp) else None
    rel_test = args.test_frac / (args.val_frac + args.test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=args.seed, stratify=strat_tmp)

    print(f"Train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    def to_loader(X_, y_, shuffle):
        ds = TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_.astype(np.float32)))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_train_loader(X_train, y_train, args.batch_size,
                                      balanced=(args.sampler == 'balanced'))
    val_loader = to_loader(X_val, y_val, False)
    test_loader = to_loader(X_test, y_test, False)

    print(f"Sampler: {args.sampler}   Loss: {args.loss}")

    model = PulseCountCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        # log_input=False: our model's output is already a rate (softplus + 1),
        # not a log-rate. full=True adds the Stirling term so the printed loss
        # approximates -log P(y|pred) rather than dropping a target-only
        # constant; it does not change the gradient w.r.t. predictions.
        criterion = nn.PoissonNLLLoss(log_input=False, full=True)

    best_val_mae = float('inf')
    ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
    history = []

    n_batches = len(train_loader)
    batch_progress_step = max(1, n_batches // 4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n += len(xb)
            if batch_idx % batch_progress_step == 0 or batch_idx == n_batches:
                print(f"    epoch {epoch:3d}/{args.epochs}  batch {batch_idx:4d}/{n_batches}",
                      flush=True)
        train_loss /= n

        val_loss, val_mae, val_acc = evaluate(model, val_loader, criterion, device)
        history.append((epoch, train_loss, val_loss, val_mae, val_acc))
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_MAE={val_mae:.3f}  val_acc={val_acc:.3f}")

        if live_plot is not None:
            live_plot.update(epoch, train_loss, val_loss, val_mae, val_acc)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({'model_state': model.state_dict(),
                        'input_len': X.shape[1],
                        'val_mae': val_mae}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_loss, test_mae, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nBest val MAE: {best_val_mae:.3f}")
    print(f"Test: loss={test_loss:.4f}  MAE={test_mae:.3f}  exact-match accuracy={test_acc:.3f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    test_preds_round = np.clip(np.round(test_preds), 1, None).astype(int)
    per_class_report(y_test, test_preds_round)

    np.savez(os.path.join(args.out_dir, 'test_split.npz'), X_test=X_test, y_test=y_test)

    with open(os.path.join(args.out_dir, 'history.csv'), 'w') as f:
        f.write('epoch,train_loss,val_loss,val_mae,val_acc\n')
        for row in history:
            f.write(','.join(str(x) for x in row) + '\n')

    print(f"\nSaved best model to {ckpt_path}")
    print(f"Saved held-out test split to {os.path.join(args.out_dir, 'test_split.npz')}")
    print("Run validate.py to generate the pulse-count distribution histogram.")

    if live_plot is not None:
        live_plot.keep_open()


if __name__ == '__main__':
    main()
