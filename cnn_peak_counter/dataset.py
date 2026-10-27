"""
Load synthetic GRB light curves produced by lc_pulse_avalanche/simulate_GRBs.py
into arrays suitable for training a 1D CNN to predict the number of parent
(avalanche-triggering) pulses in each light curve.

Each `lcN.txt` file has columns:
    times  lc  err_lc  model  modelbkg  bg  T90  n_sig_pulses  n_pls  n_parent_pulses
`n_sig_pulses` is currently always "None" (that computation path is broken
upstream in statistical_test.py) so it's ignored here.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

_COLUMNS = ['t', 'lc', 'err_lc', 'model', 'modelbkg', 'bg', 'T90',
            'n_sig_pulses', 'n_pls', 'n_parent_pulses']


def _lc_files(run_dir):
    files = glob.glob(os.path.join(run_dir, 'lc*.txt'))
    files = [f for f in files if not f.endswith('_lc_params.txt')]
    if not files:
        raise FileNotFoundError(f"No lc*.txt files found in {run_dir}")

    def idx(f):
        m = re.search(r'lc(\d+)\.txt$', os.path.basename(f))
        return int(m.group(1))

    return sorted(files, key=idx)


def find_latest_run(base_dir=None):
    """Return the most recently modified simulated_LCs/<timestamp> directory."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'simulated_LCs')
    candidates = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def load_run(run_dir, normalize=True):
    """
    Load every simulated light curve in `run_dir` into a fixed-length array,
    together with its number-of-parent-pulses label.

    Returns
    -------
    X : float32 array, shape (N, L)
        The background-subtracted count time series for each LC. If
        `normalize`, each row is divided by its own peak absolute value
        (shape-only input; the CNN never sees absolute flux).
    y : int64 array, shape (N,)
        n_parent_pulses for each LC.
    """
    files = _lc_files(run_dir)

    rows, labels = [], []
    target_len = None
    for f in files:
        df = pd.read_csv(f, sep=r'\s+', comment='#', header=None, names=_COLUMNS)
        lc = df['lc'].to_numpy(dtype=np.float32)
        n_parent_pulses = int(df['n_parent_pulses'].iloc[0])

        if target_len is None:
            target_len = len(lc)
        elif len(lc) != target_len:
            # defensive padding/truncation in case a run mixes lengths
            # (e.g. different instrument/config settings across runs)
            if len(lc) > target_len:
                lc = lc[:target_len]
            else:
                lc = np.pad(lc, (0, target_len - len(lc)))

        rows.append(lc)
        labels.append(n_parent_pulses)

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    if normalize:
        peak = np.max(np.abs(X), axis=1, keepdims=True)
        peak[peak == 0] = 1.0
        X = X / peak

    return X, y


def cap_per_class(X, y, cap, seed=42):
    """
    Undersample the dominant classes: keep at most `cap` examples of each
    label value (chosen at random), keeping every example for classes
    already at or below `cap`. Unlike WeightedRandomSampler-based balanced
    sampling (train.py's make_train_loader, which reweights how often
    existing examples are *seen* during training but leaves the underlying
    dataset untouched), this actually shrinks the training set itself --
    a different, complementary way to counter the class imbalance
    described in REPORT.md Sec. 7.3/7.6.
    """
    rng = np.random.default_rng(seed)
    keep_idx = []
    for k in np.unique(y):
        idx = np.flatnonzero(y == k)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep_idx.append(idx)
    keep_idx = np.sort(np.concatenate(keep_idx))
    return X[keep_idx], y[keep_idx]
