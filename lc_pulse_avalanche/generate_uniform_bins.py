"""
Build a genuinely uniform-by-construction dataset: exactly `--per-bin`
light curves for each n_parent_pulses value from 1 to 10 (bin 0 is skipped
-- it's unreachable: avalanche.py redraws mu_s until nonzero, and even if
it didn't, a zero-pulse candidate would almost certainly fail the T90/S-N
constraint filters anyway).

Method: mu0 is the *only* SS96 parameter that controls n_parent_pulses
(avalanche.py:generate_avalanche, mu_s ~ Poisson(mu0)); the other six
Bazzanini-calibrated parameters (mu, alpha, delta1, delta2, tau_min,
tau_max) are held fixed throughout. For each target bin k, mu0 is set to k
so that k becomes the mode of that batch's distribution, a batch is
generated, and only the light curves that land exactly on k are kept --
everything else generated in that batch is discarded. This repeats in
rounds per bin until `--per-bin` exact matches are collected (or
--max-rounds is hit).

Usage:
    conda run -n pygad3 python generate_uniform_bins.py --per-bin 300  --out-dir ../simulated_LCs/uniform_3000
    conda run -n pygad3 python generate_uniform_bins.py --per-bin 1000 --out-dir ../simulated_LCs/uniform_10000
"""
import argparse
import os
import shutil
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.append('../statistical_test')
sys.path.append('../lc_pulse_avalanche')

from statistical_test import generate_GRBs, instr_swift  # noqa: E402

# Bazzanini-calibrated Swift v36 parameters -- only mu0 varies per bin.
BASE_PARAMS = dict(mu=1.34, alpha=2.53, delta1=-0.75, delta2=0.27,
                    tau_min=0.03, tau_max=56.8)
N_CUT = 2500
T_F = 150


def generate_batch(mu0, batch_size, scratch_dir):
    """Generate `batch_size` accepted light curves at the given mu0,
    exported to `scratch_dir`. Returns the list of GRB objects (each has
    .n_parent_pulses and .name giving its exported filename)."""
    os.makedirs(scratch_dir, exist_ok=True)
    grbs = generate_GRBs(
        N_grb=batch_size,
        mu=BASE_PARAMS['mu'], mu0=mu0, alpha=BASE_PARAMS['alpha'],
        delta1=BASE_PARAMS['delta1'], delta2=BASE_PARAMS['delta2'],
        tau_min=BASE_PARAMS['tau_min'], tau_max=BASE_PARAMS['tau_max'],
        instrument='swift', bin_time=instr_swift['res'],
        eff_area=instr_swift['eff_area'], bg_level=instr_swift['bg_level'],
        t90_threshold=instr_swift['t90_threshold'],
        sn_threshold=instr_swift['sn_threshold'], t_f=T_F, filter=True,
        export_files=True, export_path=scratch_dir, n_cut=N_CUT,
        with_bg=False, remove_instrument_path=True, test_pulse_distr=False)
    return grbs


def harvest_bin(target_k, per_bin, out_dir, next_idx, batch_size=300, max_rounds=40):
    """Collect `per_bin` light curves with n_parent_pulses == target_k,
    writing them into out_dir as lc{next_idx}.txt etc. Returns the updated
    next_idx (so numbering continues across bins) and the count actually
    collected (may be < per_bin if max_rounds is hit)."""
    collected = 0
    scratch_dir = os.path.join(out_dir, '_scratch')

    for round_i in range(1, max_rounds + 1):
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)
        grbs = generate_batch(mu0=target_k, batch_size=batch_size, scratch_dir=scratch_dir)

        for grb in grbs:
            if collected >= per_bin:
                break
            if grb.n_parent_pulses != target_k:
                continue
            src_txt = os.path.join(scratch_dir, grb.name)
            src_params = os.path.join(scratch_dir, grb.name.replace('.txt', '_lc_params.txt'))
            dst_txt = os.path.join(out_dir, f'lc{next_idx}.txt')
            dst_params = os.path.join(out_dir, f'lc{next_idx}_lc_params.txt')
            if os.path.exists(src_txt):
                shutil.copyfile(src_txt, dst_txt)
            if os.path.exists(src_params):
                shutil.copyfile(src_params, dst_params)
            next_idx += 1
            collected += 1

        print(f"\r  [bin {target_k:>2d}] round {round_i:>2d}/{max_rounds}  "
              f"{collected:>4d}/{per_bin} collected", end='', flush=True)
        if collected >= per_bin:
            break

    print()
    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)
    return next_idx, collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-bin', type=int, required=True,
                         help='Target number of light curves per n_parent_pulses bin (1-10).')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=300,
                         help='Light curves generated per round while harvesting each bin.')
    parser.add_argument('--max-rounds', type=int, default=40,
                         help='Give up on a bin after this many rounds (per_bin may not be reached).')
    parser.add_argument('--max-bin', type=int, default=10,
                         help='Highest n_parent_pulses bin to harvest (inclusive). Bins above '
                              'mu0~8 take dramatically longer per accepted match (measured: '
                              '~2.5s/accepted GRB at mu0=9-10 vs ~0.4-0.9s at natural mu0=1.16, '
                              'plus a lower match rate, ~9% vs the naive 12.5% Poisson-mode '
                              'estimate) -- lower this if the high bins aren\'t worth the time.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    next_idx = 0
    summary = {}
    for k in range(1, args.max_bin + 1):
        next_idx, collected = harvest_bin(
            k, args.per_bin, args.out_dir, next_idx,
            batch_size=args.batch_size, max_rounds=args.max_rounds)
        summary[k] = collected
        if collected < args.per_bin:
            print(f"  WARNING: bin {k} only reached {collected}/{args.per_bin} "
                  f"after {args.max_rounds} rounds.")

    print("\nFinal per-bin counts:")
    for k, c in summary.items():
        print(f"  {k}: {c}/{args.per_bin}")
    print(f"\nTotal light curves written to {args.out_dir}: {next_idx}")


if __name__ == '__main__':
    main()
