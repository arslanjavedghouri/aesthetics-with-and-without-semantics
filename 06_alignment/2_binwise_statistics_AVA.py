import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


# ============================================================
# HELPER AND PLOT FUNCTIONS
# ============================================================

def load_bin_mapping(excel_file):
    """Load the Excel file and create Image ID to Bin mapping"""
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
    """
    df = pd.read_csv(summary_csv)
    df['bin'] = df['image_id'].map(bin_mapping)

    metrics = ['emd', 'norm_emd', 'bc', 'combined_euclidean', 'combined_average']
    results = {}
    for metric in metrics:
        results[f'{metric}_by_bin'] = {i: 0 for i in range(10)}

    # store per-image Yi arrays per bin (for stratified bootstrap)
    # Key: bin number -> list of 0/1 values
    results['yi_by_bin_combined_euclidean'] = {i: [] for i in range(10)}

    total_by_bin = {i: 0 for i in range(10)}

    for _, row in df.iterrows():
        bin_value = row['bin']
        if pd.notna(bin_value):
            if isinstance(bin_value, str) and bin_value.startswith('Bin-'):
                bin_value = int(bin_value.replace('Bin-', ''))
            else:
                bin_value = int(bin_value)

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

            # append Yi = 1 or 0 for combined_euclidean
            yi = 1 if row['is_close_enough_combined_euclidean'] else 0
            results['yi_by_bin_combined_euclidean'][bin_value].append(yi)

    results['total_by_bin'] = total_by_bin
    return results


def collect_all_iterations_by_bin(base_dir, bin_mapping):
    """Collect bin-wise data from all iterations for all metrics"""
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


def get_metric_config(metric):
    configs = {
        'emd':               {'title_prefix': 'EMD',                    'color': '#2E86AB', 'cmap': plt.cm.Blues},
        'norm_emd':          {'title_prefix': 'Normalized EMD',         'color': '#06A77D', 'cmap': plt.cm.Greens},
        'bc':                {'title_prefix': 'Bhattacharyya Coefficient','color': '#A23B72','cmap': plt.cm.RdPu},
        'combined_euclidean':{'title_prefix': 'Combined Euclidean',     'color': '#F18F01', 'cmap': plt.cm.Oranges},
        'combined_average':  {'title_prefix': 'Combined Average',       'color': '#C73E1D', 'cmap': plt.cm.Reds},
    }
    return configs.get(metric, configs['emd'])


def create_bin_wise_plot_for_iteration(iteration_data, output_dir, metric='emd'):
    k = iteration_data['K']
    config = get_metric_config(metric)
    data_by_bin = iteration_data[f'{metric}_by_bin']
    title = f'{config["title_prefix"]}: Images "Close Enough" (p > 0.05) by Bin at K={k}'
    color = config['color']
    filename = f'bin_wise_{metric}_K{k}.png'
    total_by_bin = iteration_data['total_by_bin']
    bin_nums = list(range(10))
    bin_labels = [f'Bin-{i}' for i in bin_nums]
    counts = [data_by_bin[i] for i in bin_nums]
    totals = [total_by_bin[i] for i in bin_nums]
    percentages = [(counts[i] / totals[i] * 100) if totals[i] > 0 else 0 for i in range(10)]
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(bin_labels))
    bars = ax.bar(x, counts, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, (bar, count, pct, total) in enumerate(zip(bars, counts, percentages, totals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{count}\n({pct:.1f}%)\n[{total}]',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Bin (AVA Rating)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of "Close Enough" Images', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    legend_text = 'Labels show:\nCount\n(Percentage)\n[Total images in bin]'
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def create_bin_wise_percentage_plot_for_iteration(iteration_data, output_dir, metric='emd'):
    k = iteration_data['K']
    config = get_metric_config(metric)
    data_by_bin = iteration_data[f'{metric}_by_bin']
    title = f'{config["title_prefix"]}: Percentage of Images "Close Enough" (p > 0.05) by Bin at K={k}'
    color = config['color']
    filename = f'bin_wise_{metric}_percentage_K{k}.png'
    total_by_bin = iteration_data['total_by_bin']
    bin_nums = list(range(10))
    bin_labels = [f'Bin-{i}' for i in bin_nums]
    counts = [data_by_bin[i] for i in bin_nums]
    totals = [total_by_bin[i] for i in bin_nums]
    percentages = [(counts[i] / totals[i] * 100) if totals[i] > 0 else 0 for i in range(10)]
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(bin_labels))
    bars = ax.bar(x, percentages, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, (bar, count, pct, total) in enumerate(zip(bars, counts, percentages, totals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{pct:.1f}%\n({count}/{total})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Bin (AVA Rating)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of "Close Enough" Images (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    legend_text = 'Labels show:\nPercentage\n(Close Enough / Total)'
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def create_all_iterations_combined_plot(all_iterations, output_dir, metric='emd'):
    config = get_metric_config(metric)
    title = f'{config["title_prefix"]}: Images "Close Enough" by Bin Across All Iterations'
    color_map = config['cmap']
    filename = f'bin_wise_{metric}_all_iterations.png'
    bin_nums = list(range(10))
    bin_labels = [f'Bin-{i}' for i in bin_nums]
    k_values = [it['K'] for it in all_iterations]
    data_matrix = []
    for iteration in all_iterations:
        data_by_bin = iteration[f'{metric}_by_bin']
        counts = [data_by_bin[i] for i in bin_nums]
        data_matrix.append(counts)
    data_matrix = np.array(data_matrix).T
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(data_matrix, aspect='auto', cmap=color_map, interpolation='nearest')
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(bin_labels)))
    ax.set_xticklabels([f'K={k}' for k in k_values], rotation=45, ha='right')
    ax.set_yticklabels(bin_labels)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Images', rotation=270, labelpad=20, fontweight='bold')
    for i in range(len(bin_labels)):
        for j in range(len(k_values)):
            ax.text(j, i, int(data_matrix[i, j]),
                    ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Iteration (K value)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bin (AVA Rating)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot: {filename}")
    plt.close()


def create_line_plot_by_bin(all_iterations, output_dir, metric='emd'):
    config = get_metric_config(metric)
    title = f'{config["title_prefix"]}: Trend of "Close Enough" Images by Bin Across Iterations'
    filename = f'bin_wise_{metric}_trends.png'
    colors = config['cmap'](np.linspace(0.3, 1, 10))
    bin_nums = list(range(10))
    bin_labels = [f'Bin-{i}' for i in bin_nums]
    k_values = [it['K'] for it in all_iterations]
    fig, ax = plt.subplots(figsize=(16, 10))
    for i, (bin_num, bin_label) in enumerate(zip(bin_nums, bin_labels)):
        counts = [iteration[f'{metric}_by_bin'][bin_num] for iteration in all_iterations]
        ax.plot(k_values, counts, marker='o', linewidth=2.5, markersize=8,
                label=bin_label, color=colors[i], alpha=0.8)
    ax.set_xlabel('K (Number of Resamples)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of "Close Enough" Images', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved trend plot: {filename}")
    plt.close()


def create_summary_table(all_iterations, output_dir):
    metrics = ['emd', 'norm_emd', 'bc', 'combined_euclidean', 'combined_average']
    for metric in metrics:
        rows = []
        for iteration in all_iterations:
            k = iteration['K']
            data_by_bin = iteration[f'{metric}_by_bin']
            row = {'K': k}
            for i in range(10):
                row[f'Bin-{i}'] = data_by_bin[i]
            rows.append(row)
        df = pd.DataFrame(rows)
        filename = f'bin_wise_{metric}_summary.csv'
        output_file = os.path.join(output_dir, filename)
        df.to_csv(output_file, index=False)
        print(f"Saved summary table: {filename}")


def create_metric_comparison_plot(all_iterations, output_dir, k_value):
    iteration = next((it for it in all_iterations if it['K'] == k_value), None)
    if iteration is None:
        print(f"Warning: K={k_value} not found, skipping comparison plot")
        return
    metrics = ['emd', 'norm_emd', 'bc', 'combined_euclidean', 'combined_average']
    metric_labels = ['EMD', 'Norm EMD', 'BC', 'Comb Eucl', 'Comb Avg']
    colors = ['#2E86AB', '#06A77D', '#A23B72', '#F18F01', '#C73E1D']
    bin_nums = list(range(10))
    bin_labels = [f'Bin-{i}' for i in bin_nums]
    fig, ax = plt.subplots(figsize=(18, 10))
    x = np.arange(len(bin_labels))
    width = 0.15
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        data_by_bin = iteration[f'{metric}_by_bin']
        counts = [data_by_bin[b] for b in bin_nums]
        offset = width * (i - 2)
        bars = ax.bar(x + offset, counts, width, label=label, color=color, alpha=0.8, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xlabel('Bin (AVA Rating)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of "Close Enough" Images', fontsize=14, fontweight='bold')
    ax.set_title(f'Comparison of All 5 Metrics by Bin at K={k_value}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.legend(fontsize=11, loc='best', ncol=5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    filename = f'metric_comparison_K{k_value}.png'
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved metric comparison plot: {filename}")
    plt.close()


def create_all_metrics_trend_plot(all_iterations, output_dir, bin_num):
    metrics = ['emd', 'norm_emd', 'bc', 'combined_euclidean', 'combined_average']
    metric_labels = ['EMD', 'Normalized EMD', 'BC', 'Combined Euclidean', 'Combined Average']
    colors = ['#2E86AB', '#06A77D', '#A23B72', '#F18F01', '#C73E1D']
    k_values = [it['K'] for it in all_iterations]
    fig, ax = plt.subplots(figsize=(14, 8))
    for metric, label, color in zip(metrics, metric_labels, colors):
        counts = [iteration[f'{metric}_by_bin'][bin_num] for iteration in all_iterations]
        ax.plot(k_values, counts, marker='o', linewidth=2.5, markersize=8,
                label=label, color=color, alpha=0.8)
    ax.set_xlabel('K (Number of Resamples)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of "Close Enough" Images', fontsize=14, fontweight='bold')
    ax.set_title(f'All Metrics Trend for Bin-{bin_num} Across Iterations', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'all_metrics_trend_bin{bin_num}.png'
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved all-metrics trend plot for Bin-{bin_num}: {filename}")
    plt.close()


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def run_statistical_analysis(iteration_data, output_dir, k, N_BOOT=10000):
    """
    Statistical analysis of the within-baseline rates across bins:
      1. Stratified nonparametric bootstrap CIs (per-image Yi resampled within bins)
      2. Binomial logistic regression: linear vs quadratic
      3. Likelihood-ratio test for U-shaped pattern
      4. Planned contrast: extreme vs middle bins with bootstrap CI
      5. Figure saved to output_dir

    Uses the combined_euclidean metric (the one reported in the paper).
    Yi = 1 if image is 'close enough', 0 otherwise - taken directly from per-image data.
    """

    print(f"\n{'=' * 65}")
    print(f"STATISTICAL ANALYSIS - Combined Euclidean metric, K={k}")
    print(f"{'=' * 65}")

    # extract counts from real per-image Yi values
    yi_by_bin   = iteration_data['yi_by_bin_combined_euclidean']  # dict: bin -> list of 0/1
    total_by_bin = iteration_data['total_by_bin']

    bin_nums    = list(range(10))
    bin_labels  = [f'Bin-{i+1}' for i in bin_nums]   # bins labelled 1-10

    close_arr = np.array([sum(yi_by_bin[i])         for i in bin_nums])
    total_arr = np.array([len(yi_by_bin[i])         for i in bin_nums])
    pct       = np.where(total_arr > 0, close_arr / total_arr * 100, 0.0)

    # 1. STRATIFIED NONPARAMETRIC BOOTSTRAP
    # Resample WITH REPLACEMENT from actual Yi values within each bin.
    # bootstrap on the per-image 0/1 values
    rng = np.random.default_rng(42)
    boot_pct = np.zeros((N_BOOT, 10))

    for b in range(N_BOOT):
        for i in bin_nums:
            yi = np.array(yi_by_bin[i])
            if len(yi) > 0:
                sample = rng.choice(yi, size=len(yi), replace=True)
                boot_pct[b, i] = sample.mean() * 100
            else:
                boot_pct[b, i] = 0.0

    boot_ci_lo = np.percentile(boot_pct, 2.5,  axis=0)
    boot_ci_hi = np.percentile(boot_pct, 97.5, axis=0)

    print(f"\nStratified nonparametric bootstrap (N={N_BOOT:,}) - per-bin 95% CIs:")
    for i, (bl, p, lo, hi, nc, nt) in enumerate(
            zip(bin_labels, pct, boot_ci_lo, boot_ci_hi, close_arr, total_arr)):
        print(f"  {bl}: {p:.1f}%  [{lo:.1f}, {hi:.1f}]  ({nc}/{nt})")

    # 2. BINOMIAL LOGISTIC REGRESSION
    z      = np.arange(1, 11) - 5.5    # centered bin variable
    z2     = z ** 2
    endog  = np.column_stack([close_arr, total_arr - close_arr])

    res_lin  = sm.GLM(endog,
                      sm.add_constant(z),
                      family=sm.families.Binomial()).fit()
    res_quad = sm.GLM(endog,
                      sm.add_constant(np.column_stack([z, z2])),
                      family=sm.families.Binomial()).fit()

    # 3. LIKELIHOOD-RATIO TEST
    lrt_stat = 2 * (res_quad.llf - res_lin.llf)
    lrt_p    = 1 - chi2.cdf(lrt_stat, df=1)
    b2       = res_quad.params[2]
    b2_ci    = res_quad.conf_int()
    b2_lo    = float(b2_ci[2][0])
    b2_hi    = float(b2_ci[2][1])

    print(f"\nBinomial logistic regression:")
    print(f"  LRT χ²(1) = {lrt_stat:.3f},  p = {lrt_p:.2e}")
    print(f"  β₂ (quadratic) = {b2:.4f},  95% CI [{b2_lo:.4f}, {b2_hi:.4f}]")
    if lrt_p < 0.05 and b2 > 0:
        print(f"  Significant positive quadratic -> U-shaped pattern confirmed")

    # 4. PLANNED CONTRAST: extreme vs middle bins
    # Extreme: Bin-1, Bin-2, Bin-8, Bin-9, Bin-10 -> indices 0,1,7,8,9
    # Middle:  Bin-4, Bin-5, Bin-6               -> indices 3,4,5
    idx_ext = [0, 1, 7, 8, 9]
    idx_mid = [3, 4, 5]

    ext_close = close_arr[idx_ext].sum(); ext_total = total_arr[idx_ext].sum()
    mid_close = close_arr[idx_mid].sum(); mid_total = total_arr[idx_mid].sum()
    ext_pct = pct[idx_ext].mean()
    mid_pct = pct[idx_mid].mean()
    obs_diff  = ext_pct - mid_pct

    # bootstrap the extreme-middle difference using the same stratified resamples
    boot_ext  = boot_pct[:, idx_ext].mean(axis=1)   # mean pct over extreme bins
    boot_mid  = boot_pct[:, idx_mid].mean(axis=1)
    boot_diff = boot_ext - boot_mid
    diff_lo, diff_hi = np.percentile(boot_diff, [2.5, 97.5])
    p_diff = np.mean(boot_diff <= 0)

    print(f"\nPlanned contrast - extreme vs middle bins:")
    print(f"  Extreme (Bins 1-2, 8-10): {ext_pct:.1f}%  ({ext_close}/{ext_total})")
    print(f"  Middle  (Bins 4-6):       {mid_pct:.1f}%  ({mid_close}/{mid_total})")
    print(f"  Difference: {obs_diff:.1f} pp,  95% CI [{diff_lo:.1f}, {diff_hi:.1f}]")
    print(f"  p (one-sided) < {max(p_diff, 1/N_BOOT):.4f}")

    # 5. FITTED QUADRATIC CURVE
    z_fine      = np.linspace(-4.5, 4.5, 300)
    X_fine      = sm.add_constant(np.column_stack([z_fine, z_fine ** 2]))
    curve_pct   = 1 / (1 + np.exp(-(X_fine @ res_quad.params))) * 100
    x_fine_plot = z_fine + 5.5 - 1

    # 6. FIGURE
    mpl.rcParams.update({
        'font.family':       'sans-serif',
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

    x_pos = np.arange(len(bin_labels))
    fig, ax = plt.subplots(figsize=(7.1, 4.5))

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

    for xi, p, hi, nc, nt in zip(x_pos, pct, boot_ci_hi, close_arr, total_arr):
        ax.text(xi, hi + 1.5,
                f'{p:.1f}%\n({nc}/{nt})',
                ha='center', va='bottom',
                fontsize=7.2, color='#222222', linespacing=1.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('Aesthetic rating bin (AVA)', labelpad=6)
    ax.set_ylabel('Images within human baseline (%)', labelpad=6)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc='lower center', framealpha=0.85,
              edgecolor='#CCCCCC', fontsize=9, ncol=3)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6,
                  color='#BBBBBB', alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    plt.tight_layout(pad=0.8)
    fig_path = os.path.join(output_dir, f'AVA_alignment_with_stats_K{k}.png')
    pdf_path = os.path.join(output_dir, f'AVA_alignment_with_stats_K{k}.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path,           bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure saved: AVA_alignment_with_stats_K{k}.png / .pdf")


    return {
        'lrt_stat': lrt_stat, 'lrt_p': lrt_p,
        'b2': b2, 'b2_lo': b2_lo, 'b2_hi': b2_hi,
        'ext_pct': ext_pct, 'mid_pct': mid_pct,
        'obs_diff': obs_diff, 'diff_lo': diff_lo, 'diff_hi': diff_hi,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    EXCEL_FILE = r"results/combined-human-llm-rating-AVA.xlsx"   # from 06_alignment/0_build_alignment_table_AVA.py
    BASE_DIR   = r"results/split_half/AVA"
    OUTPUT_DIR = r"results/figures/AVA"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("BIN-WISE ANALYSIS ACROSS ITERATIONS - AVA DATASET (10 BINS)")
    print("WITH ALL 5 METRICS + STATISTICAL ANALYSIS (Combined Euclidean)")
    print("=" * 70)

    bin_mapping    = load_bin_mapping(EXCEL_FILE)
    all_iterations = collect_all_iterations_by_bin(BASE_DIR, bin_mapping)

    if len(all_iterations) == 0:
        print("\nNo iteration data found!")
    else:
        print(f"\nFound {len(all_iterations)} iterations")
        k_values = [it['K'] for it in all_iterations]
        print(f"K values: {k_values}")

        metrics = ['emd', 'norm_emd', 'bc', 'combined_euclidean', 'combined_average']

        print("\nCreating individual iteration plots (count-based)...")
        for iteration in all_iterations:
            k = iteration['K']
            print(f"\nK={k}:")
            for metric in metrics:
                create_bin_wise_plot_for_iteration(iteration, OUTPUT_DIR, metric=metric)

        print("\nCreating individual iteration plots (percentage-based)...")
        for iteration in all_iterations:
            k = iteration['K']
            print(f"\nK={k}:")
            for metric in metrics:
                create_bin_wise_percentage_plot_for_iteration(iteration, OUTPUT_DIR, metric=metric)

        print("\nCreating combined heatmap plots...")
        for metric in metrics:
            create_all_iterations_combined_plot(all_iterations, OUTPUT_DIR, metric=metric)

        print("\nCreating trend line plots...")
        for metric in metrics:
            create_line_plot_by_bin(all_iterations, OUTPUT_DIR, metric=metric)

        print("\nCreating metric comparison plots...")
        for iteration in all_iterations:
            create_metric_comparison_plot(all_iterations, OUTPUT_DIR, iteration['K'])

        print("\nCreating all-metrics trend plots for each bin...")
        for bin_num in range(10):
            create_all_metrics_trend_plot(all_iterations, OUTPUT_DIR, bin_num)

        print("\nCreating summary tables...")
        create_summary_table(all_iterations, OUTPUT_DIR)

        # run the statistical analysis for the largest K
        print("\n" + "=" * 70)
        print("RUNNING STATISTICAL ANALYSIS (largest K value)")
        print("=" * 70)
        largest_k_iteration = max(all_iterations, key=lambda x: x['K'])
        run_statistical_analysis(
            largest_k_iteration,
            OUTPUT_DIR,
            k=largest_k_iteration['K'],
            N_BOOT=10000
        )

        print("\n" + "=" * 70)
        print("ALL DONE")
        print("=" * 70)