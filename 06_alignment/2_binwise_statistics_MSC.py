"""
MSC LLM-Human Alignment - Complete Analysis
============================================
Adapted from AVA analysis script for MSC dataset (5 bins instead of 10).

Key differences from AVA version:
  - 5 aesthetic bins (Bin-1 to Bin-5) instead of 10
  - Planned contrast: high bins (4-5) vs low bins (1-2)
  - Logistic regression tests linear vs quadratic (expects monotone increasing)
  - Rating scale 1-5 instead of 1-10
  - No challenge metadata

Output files
------------
  MSC_alignment_pct_only.png       - percentage only, no counts (300 dpi)
  MSC_alignment_pct_only.pdf       - vector version
  MSC_alignment_with_counts.png    - percentage + counts (300 dpi)
  MSC_alignment_with_counts.pdf    - vector version
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

sns.set_style("whitegrid")

# ============================================================
# PATHS - update if needed
# ============================================================
EXCEL_FILE = r"results/MSC_LLM_ratings.xlsx"   # from 05_llm_rating/rate_captions_MSC.py
BASE_DIR   = r"results/split_half/MSC"
OUTPUT_DIR = r"results/figures/MSC"

N_BINS = 5   # MSC has 5 bins

# ============================================================
# DATA LOADING
# ============================================================
def load_bin_mapping(excel_file):
    """Load the Excel file and create Image ID to Bin mapping."""
    print(f"Loading bin mapping from: {excel_file}")
    df = pd.read_excel(excel_file)
    bin_mapping = {}
    for _, row in df.iterrows():
        image_id = row['Image ID']
        bin_value = row['Bin']
        bin_mapping[image_id] = bin_value
    print(f"Loaded {len(bin_mapping)} image-to-bin mappings")
    return bin_mapping


def analyze_iteration_by_bin(summary_csv, bin_mapping):
    """
    Analyze a single iteration's summary results by bin.
    Returns counts of 'close enough' images per bin for all 5 metrics.
    Also returns per-image binary arrays per bin (for stratified bootstrap).
    MSC: bins 1-5 (stored as indices 1-5, not 0-based like AVA).
    """
    df = pd.read_csv(summary_csv)
    df['bin'] = df['image_id'].map(bin_mapping)

    metrics = ['emd', 'norm_emd', 'bc', 'combined_euclidean', 'combined_average']
    results = {}
    for metric in metrics:
        results[f'{metric}_by_bin'] = {i: 0 for i in range(1, N_BINS + 1)}

    # Per-image binary arrays for stratified bootstrap
    results['yi_by_bin_combined_euclidean'] = {i: [] for i in range(1, N_BINS + 1)}
    total_by_bin = {i: 0 for i in range(1, N_BINS + 1)}

    for _, row in df.iterrows():
        bin_value = row['bin']
        if pd.notna(bin_value):
            if isinstance(bin_value, str) and bin_value.startswith('Bin-'):
                bin_value = int(bin_value.replace('Bin-', ''))
            else:
                bin_value = int(bin_value)

            if bin_value not in range(1, N_BINS + 1):
                continue

            total_by_bin[bin_value] += 1

            if row['is_close_enough_emd']:
                results['emd_by_bin'][bin_value] += 1
            if row['is_close_enough_norm_emd']:
                results['norm_emd_by_bin'][bin_value] += 1
            if row['is_close_enough_bc']:
                results['bc_by_bin'][bin_value] += 1
            if row['is_close_enough_combined_euclidean']:
                results['combined_euclidean_by_bin'][bin_value] += 1
            if row['is_close_enough_combined_average']:
                results['combined_average_by_bin'][bin_value] += 1

            yi = 1 if row['is_close_enough_combined_euclidean'] else 0
            results['yi_by_bin_combined_euclidean'][bin_value].append(yi)

    results['total_by_bin'] = total_by_bin
    return results


def collect_all_iterations_by_bin(base_dir, bin_mapping):
    """Collect bin-wise data from all iterations."""
    results = []
    base_path = Path(base_dir)
    for folder in sorted(base_path.glob('*-iterations*')):
        summary_file = folder / 'summary_results.csv'
        if summary_file.exists():
            k_match = folder.name.split('-')[0]
            try:
                k = int(k_match)
            except:
                continue
            print(f"Processing K={k}: {summary_file}")
            iteration_results = analyze_iteration_by_bin(summary_file, bin_mapping)
            iteration_results['K'] = k
            results.append(iteration_results)
    return sorted(results, key=lambda x: x['K'])


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
def run_statistical_analysis(iteration_data, output_dir, k, N_BOOT=10000):
    """
    Full statistical analysis for MSC (5 bins):
      1. Stratified nonparametric bootstrap CIs
      2. Binomial logistic regression: linear vs quadratic
      3. Likelihood-ratio test
      4. Planned contrast: high bins (4-5) vs low bins (1-2)
      5. Global overall % within baseline (collapsing all bins)
      6. Class-balanced global % (robustness check)
      7. Figure
    """
    print(f"\n{'=' * 65}")
    print(f"STATISTICAL ANALYSIS - MSC, Combined Euclidean metric, K={k}")
    print(f"{'=' * 65}")

    bin_nums  = list(range(1, N_BINS + 1))       # [1, 2, 3, 4, 5]
    bin_labels = [f'Bin-{i}' for i in bin_nums]

    yi_by_bin    = iteration_data['yi_by_bin_combined_euclidean']
    total_by_bin = iteration_data['total_by_bin']

    close_arr = np.array([sum(yi_by_bin[i])  for i in bin_nums])
    total_arr = np.array([len(yi_by_bin[i])  for i in bin_nums])
    pct       = np.where(total_arr > 0, close_arr / total_arr * 100, 0.0)

    # 1. STRATIFIED NONPARAMETRIC BOOTSTRAP
    rng      = np.random.default_rng(42)
    boot_pct = np.zeros((N_BOOT, N_BINS))

    for b in range(N_BOOT):
        for idx, i in enumerate(bin_nums):
            yi = np.array(yi_by_bin[i])
            if len(yi) > 0:
                sample = rng.choice(yi, size=len(yi), replace=True)
                boot_pct[b, idx] = sample.mean() * 100
            else:
                boot_pct[b, idx] = 0.0

    boot_ci_lo = np.percentile(boot_pct,  2.5, axis=0)
    boot_ci_hi = np.percentile(boot_pct, 97.5, axis=0)

    print(f"\nStratified nonparametric bootstrap (N={N_BOOT:,}) - per-bin 95% CIs:")
    for bl, p, lo, hi, nc, nt in zip(bin_labels, pct, boot_ci_lo,
                                       boot_ci_hi, close_arr, total_arr):
        print(f"  {bl}: {p:.1f}%  [{lo:.1f}, {hi:.1f}]  ({nc}/{nt})")

    # 2. BINOMIAL LOGISTIC REGRESSION
    # Centered on bin 3 (midpoint of 1-5)
    z     = np.array(bin_nums) - 3.0    # [-2, -1, 0, 1, 2]
    z2    = z ** 2
    endog = np.column_stack([close_arr, total_arr - close_arr])

    res_lin  = sm.GLM(endog,
                      sm.add_constant(z),
                      family=sm.families.Binomial()).fit()
    res_quad = sm.GLM(endog,
                      sm.add_constant(np.column_stack([z, z2])),
                      family=sm.families.Binomial()).fit()

    # 3. LIKELIHOOD-RATIO TEST
    lrt_stat = 2 * (res_quad.llf - res_lin.llf)
    lrt_p    = 1 - chi2.cdf(lrt_stat, df=1)
    b1       = res_quad.params[1]   # linear term (dominant for MSC)
    b2       = res_quad.params[2]   # quadratic term
    b1_ci    = res_quad.conf_int()
    b1_lo    = float(b1_ci[1][0])
    b1_hi    = float(b1_ci[1][1])
    b2_lo    = float(b1_ci[2][0])
    b2_hi    = float(b1_ci[2][1])

    print(f"\nBinomial logistic regression:")
    print(f"  LRT χ²(1) = {lrt_stat:.3f},  p = {lrt_p:.2e}")
    print(f"  β₁ (linear)    = {b1:.4f},  95% CI [{b1_lo:.4f}, {b1_hi:.4f}]")
    print(f"  β₂ (quadratic) = {b2:.4f},  95% CI [{b2_lo:.4f}, {b2_hi:.4f}]")

    # Fitted curve
    z_fine      = np.linspace(-2, 2, 300)
    X_fine      = sm.add_constant(np.column_stack([z_fine, z_fine ** 2]))
    curve_pct   = 1 / (1 + np.exp(-(X_fine @ res_quad.params))) * 100
    x_fine_plot = z_fine + 3.0 - 1   # shift to 0-indexed x positions

    # 4. PLANNED CONTRAST: high (4-5) vs low (1-2)
    # indices in 0-based boot_pct array: bin 1->idx 0, bin 4->idx 3, bin 5->idx 4
    idx_high = [3, 4]   # bins 4 and 5
    idx_low  = [0, 1]   # bins 1 and 2

    high_close = close_arr[idx_high].sum(); high_total = total_arr[idx_high].sum()
    low_close  = close_arr[idx_low].sum();  low_total  = total_arr[idx_low].sum()
    high_pct = pct[idx_high].mean()
    low_pct = pct[idx_low].mean()
    obs_diff = high_pct - low_pct

    boot_high = boot_pct[:, idx_high].mean(axis=1)
    boot_low  = boot_pct[:, idx_low].mean(axis=1)
    boot_diff = boot_high - boot_low
    diff_lo, diff_hi = np.percentile(boot_diff, [2.5, 97.5])
    p_diff = np.mean(boot_diff <= 0)

    print(f"\nPlanned contrast - high (Bins 4-5) vs low (Bins 1-2):")
    print(f"  High bins (4-5): {high_pct:.1f}%  ({high_close}/{high_total})")
    print(f"  Low bins  (1-2): {low_pct:.1f}%   ({low_close}/{low_total})")
    print(f"  Difference: {obs_diff:.1f} pp,  95% CI [{diff_lo:.1f}, {diff_hi:.1f}]")
    print(f"  p (one-sided) < {max(p_diff, 1/N_BOOT):.4f}")

    # 5. GLOBAL OVERALL %
    global_close = close_arr.sum()
    global_total = total_arr.sum()
    global_pct   = global_close / global_total * 100

    # Bootstrap CI for global %
    boot_global = boot_pct.mean(axis=1)   # mean across bins per bootstrap sample
    # Alternative: use raw counts for unweighted global
    boot_global_raw = np.array([
        sum(boot_pct[b, idx] * total_arr[idx] for idx in range(N_BINS)) / global_total
        for b in range(N_BOOT)
    ])
    global_ci_lo, global_ci_hi = np.percentile(boot_global_raw, [2.5, 97.5])

    print(f"\nGlobal overall % within baseline (MSC, all bins):")
    print(f"  {global_pct:.1f}%  ({global_close}/{global_total})")
    print(f"  95% bootstrap CI: [{global_ci_lo:.1f}, {global_ci_hi:.1f}]")

    # 6. CLASS-BALANCED GLOBAL % (robustness check)
    # Simple mean of per-bin percentages (equal weight to each bin)
    balanced_pct    = pct.mean()
    boot_balanced   = boot_pct.mean(axis=1)
    bal_ci_lo, bal_ci_hi = np.percentile(boot_balanced, [2.5, 97.5])

    print(f"\nClass-balanced global % (mean of per-bin %):")
    print(f"  {balanced_pct:.1f}%")
    print(f"  95% bootstrap CI: [{bal_ci_lo:.1f}, {bal_ci_hi:.1f}]")

    # 7. FIGURE
    mpl.rcParams.update({
        'font.family':      'sans-serif',
        'font.size':         10,
        'font.weight':       'normal',
        'axes.labelsize':    11,
        'axes.labelweight':  'normal',
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
    })

    BAR_COLOR   = '#8B2635'
    ERR_COLOR   = '#333333'
    CURVE_COLOR = '#E69F00'

    x_pos = np.arange(N_BINS)

    def make_figure(show_counts):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        ax.bar(x_pos, pct, color=BAR_COLOR, width=0.62,
               zorder=2, linewidth=0, label='% within baseline')

        ax.errorbar(x_pos, pct,
                    yerr=[pct - boot_ci_lo, boot_ci_hi - pct],
                    fmt='none', color=ERR_COLOR,
                    capsize=4, capthick=1.3, elinewidth=1.3,
                    zorder=3, label='95% bootstrap CI')

        ax.plot(x_fine_plot, curve_pct, '-',
                color=CURVE_COLOR, linewidth=2.0, zorder=4,
                label='Fitted quadratic logistic curve')

        for xi, p, hi, nc, nt in zip(x_pos, pct, boot_ci_hi,
                                      close_arr, total_arr):
            label_txt = f'{p:.1f}%\n({nc}/{nt})' if show_counts else f'{p:.1f}%'
            ax.text(xi, hi + 1.5, label_txt,
                    ha='center', va='bottom',
                    fontsize=8, color='#222222', linespacing=1.3, zorder=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel('Aesthetic rating Classes (MSC)', labelpad=6)
        ax.set_ylabel('Images within human baseline (%)', labelpad=6)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        ax.legend(loc='upper left', framealpha=0.85,
                  edgecolor='#CCCCCC', fontsize=7.5, ncol=1)

        ax.yaxis.grid(True, linestyle='--', linewidth=0.6,
                      color='#BBBBBB', alpha=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        plt.tight_layout(pad=0.8)
        return fig

    os.makedirs(output_dir, exist_ok=True)

    fig1 = make_figure(show_counts=False)
    fig1.savefig(os.path.join(output_dir, f'MSC_alignment_pct_only_K{k}.png'),
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig(os.path.join(output_dir, f'MSC_alignment_pct_only_K{k}.pdf'),
                 bbox_inches='tight', facecolor='white')
    fig1.savefig(os.path.join(output_dir, f'MSC_alignment_pct_only_K{k}.svg'),
                 bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"\nFigure saved: MSC_alignment_pct_only_K{k}.png / .pdf / .svg")

    fig2 = make_figure(show_counts=True)
    fig2.savefig(os.path.join(output_dir, f'MSC_alignment_with_counts_K{k}.png'),
                 dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(os.path.join(output_dir, f'MSC_alignment_with_counts_K{k}.pdf'),
                 bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Figure saved: MSC_alignment_with_counts_K{k}.png / .pdf")


    return {
        'global_pct': global_pct, 'global_ci_lo': global_ci_lo,
        'global_ci_hi': global_ci_hi,
        'balanced_pct': balanced_pct, 'bal_ci_lo': bal_ci_lo,
        'bal_ci_hi': bal_ci_hi,
        'lrt_stat': lrt_stat, 'lrt_p': lrt_p,
        'b1': b1, 'b1_lo': b1_lo, 'b1_hi': b1_hi,
        'high_pct': high_pct, 'low_pct': low_pct,
        'obs_diff': obs_diff, 'diff_lo': diff_lo, 'diff_hi': diff_hi,
        'close_arr': close_arr, 'total_arr': total_arr,
        'pct': pct, 'boot_pct': boot_pct,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("BIN-WISE ANALYSIS - MSC DATASET (5 BINS)")
    print("WITH STATISTICAL ANALYSIS (Combined Euclidean metric)")
    print("=" * 70)

    bin_mapping    = load_bin_mapping(EXCEL_FILE)
    all_iterations = collect_all_iterations_by_bin(BASE_DIR, bin_mapping)

    if len(all_iterations) == 0:
        print("\nNo iteration data found. Check BASE_DIR path.")
    else:
        print(f"\nFound {len(all_iterations)} iterations")
        k_values = [it['K'] for it in all_iterations]
        print(f"K values: {k_values}")

        # Run statistical analysis on the largest K
        largest_k_iteration = max(all_iterations, key=lambda x: x['K'])
        msc_stats = run_statistical_analysis(
            largest_k_iteration,
            OUTPUT_DIR,
            k=largest_k_iteration['K'],
            N_BOOT=10000
        )

        print("\n" + "=" * 70)
        print("ALL DONE")
        print("=" * 70)