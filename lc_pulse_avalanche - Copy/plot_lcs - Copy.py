"""
Plot the observed time series for simulated light curves in a
simulated_LCs/<timestamp>/ run directory, saving one PNG per light curve
alongside its lcN.txt / lcN_lc_params.txt files.

Usage:
    conda run -n pygad3 python plot_lcs.py
    conda run -n pygad3 python plot_lcs.py --run-dir ../simulated_LCs/<timestamp>
    conda run -n pygad3 python plot_lcs.py --limit 20
"""
import argparse
import glob
import os
import re
import sys

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Same categorical palette used elsewhere in this project
# (see the `dataviz` design skill / cnn_peak_counter/validate.py).
COLOR_LC = "#22009e"
COLOR_PARENT = '#e34948'  # red -- parent pulses
COLOR_CHILD = '#eb6834'   # orange -- child pulses
COLOR_INK = '#0b0b0b'
COLOR_SUBTLE = '#52514e'
COLOR_MUTED = '#898781'
COLOR_GRID = '#e1e0d9'
COLOR_BASELINE = '#c3c2b7'
COLOR_SURFACE = '#fcfcfb'

_COLUMNS = ['t', 'lc', 'err_lc', 'model', 'modelbkg', 'bg', 'T90',
            'n_sig_pulses', 'n_pls', 'n_parent_pulses']
_PARAM_COLUMNS = ['norm', 't_delay', 'tau', 'tau_r', 'counts_pulse',
                  'is_parent', 'parent_id']


def _load_pulse_markers(lc_path, times, counts):
    """
    Read the companion lcN_lc_params.txt (ground-truth pulse list from the
    simulator). Returns a DataFrame with columns [t_delay, y, is_parent,
    parent_id], where `y` is the observed 'lc' value nearest each pulse's
    t_delay, and the row index equals each pulse's own id (rows are
    written in the same append order avalanche.py assigns as `own_id`, so
    `df.loc[parent_id]` looks up a pulse's immediate progenitor directly).
    Returns None if the params file is missing or predates the
    `is_parent`/`parent_id` columns (older datasets).
    """
    params_path = lc_path[:-4] + '_lc_params.txt'
    if not os.path.exists(params_path):
        return None

    df = pd.read_csv(params_path, sep=r'\s+', comment='#', header=None)
    if df.shape[1] < len(_PARAM_COLUMNS):
        return None  # older dataset, generated before this tracking existed
    df.columns = _PARAM_COLUMNS

    idx = np.searchsorted(times, df['t_delay'].to_numpy())
    idx = np.clip(idx, 0, len(times) - 1)
    df['y'] = counts[idx]
    df['is_parent'] = df['is_parent'].astype(bool)
    df['parent_id'] = df['parent_id'].astype(int)
    return df


def find_latest_run(base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'simulated_LCs')
    candidates = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _lc_files(run_dir):
    files = glob.glob(os.path.join(run_dir, 'lc*.txt'))
    files = [f for f in files if not f.endswith('_lc_params.txt')]

    def idx(f):
        m = re.search(r'lc(\d+)\.txt$', os.path.basename(f))
        return int(m.group(1))

    return sorted(files, key=idx)


def plot_one(lc_path, out_path, show_title=True, line_color=None, ylim=None):
    df = pd.read_csv(lc_path, sep=r'\s+', comment='#', header=None, names=_COLUMNS)
    times = df['t'].to_numpy()
    counts = df['lc'].to_numpy()
    n_parent = int(df['n_parent_pulses'].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    ax.plot(times, counts, color=line_color or COLOR_LC, linewidth=0.8, zorder=2)
    if ylim is not None:
        ax.set_ylim(*ylim)

    markers = _load_pulse_markers(lc_path, times, counts)
    if markers is not None:
        children = markers[~markers['is_parent']]
        parents = markers[markers['is_parent']]
        if len(children):
            ax.scatter(children['t_delay'], children['y'], color=COLOR_MUTED,
                       s=8, alpha=0.4, linewidths=0, zorder=3, label='Child pulse')
        if len(parents):
            ax.scatter(parents['t_delay'], parents['y'], color=COLOR_PARENT,
                       s=70, alpha=1.0, edgecolors=COLOR_INK, linewidths=1.2,
                       zorder=5, label='Parent pulse')
            for i, (_, prow) in enumerate(parents.iterrows(), start=1):
                ax.annotate(str(i), (prow['t_delay'], prow['y']), textcoords='offset points',
                            xytext=(0, 11), ha='center', va='center', fontsize=10,
                            color=COLOR_INK, fontweight='bold', zorder=6,
                            bbox=dict(boxstyle='circle,pad=0.25', facecolor='white',
                                      edgecolor=COLOR_PARENT, linewidth=1.2))

    ax.set_xlabel('Time [s]', color=COLOR_INK, fontsize=20, fontweight='bold')
    # For Swift, the 'lc' column is a count RATE (cnt/s), not raw counts --
    # see avalanche.py:506 ("the variable `_plot_lc` contains the COUNT
    # RATES, and not the counts!"). This contradicts a comment in
    # statistical_test.py's GRB construction that says the opposite; the
    # avalanche.py comment is correct since it's the code that actually
    # computes the quantity (self._model_rate = self._model / self._res).
    # (BATSE's 'lc' column genuinely is counts, not a rate -- different
    # branch, different units.)
    ax.set_ylabel('Count rate [cnt/s]', color=COLOR_INK, fontsize=20, fontweight='bold')
    if show_title:
        ax.set_title(f"{os.path.basename(lc_path)}   --   n_parent_pulses = {n_parent}",
                     color=COLOR_INK, fontsize=14, fontweight='bold', loc='left')

    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        if name in ('top', 'right'):
            spine.set_visible(False)
        else:
            spine.set_color(COLOR_INK)
            spine.set_linewidth(1.8)
    ax.tick_params(colors=COLOR_INK, labelsize=11, width=1.5)

    if markers is not None and (len(parents) or len(children)):
        ax.legend(frameon=False, labelcolor=COLOR_SUBTLE, loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=None,
                         help='Path to a simulated_LCs/<timestamp> directory. '
                              'Defaults to the most recently generated run.')
    parser.add_argument('--limit', type=int, default=None,
                         help='Only plot the first N light curves (default: all).')
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run()
    files = _lc_files(run_dir)
    if args.limit is not None:
        files = files[:args.limit]

    print(f"Plotting {len(files)} light curves from: {run_dir}")
    width = len(str(len(files))) if files else 1
    for i, f in enumerate(files, start=1):
        out_path = f[:-4] + '_plot.png'  # lcN.txt -> lcN_plot.png
        plot_one(f, out_path)
        print(f"\r  [plot]  {i:>{width}d}/{len(files)}", end='', flush=True)
    print()
    print(f"Saved plots alongside each lcN.txt in: {run_dir}")


if __name__ == '__main__':
    main()
