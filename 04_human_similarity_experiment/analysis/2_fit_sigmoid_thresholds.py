import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# Okabe-Ito colours
COLOR_CURVE     = "#0072B2"   # sigmoid fit
COLOR_DATA      = "#D55E00"   # data points
COLOR_THRESHOLD = "#E69F00"   # 75% threshold

plt.rcParams.update({
    'font.size':         14,
    'axes.titlesize':    14,
    'axes.labelsize':    14,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   9,
    'figure.titlesize':  14,
    'font.family':       'sans-serif',
})

# ============================================================
# CONFIG
# ============================================================
# trials of all participants: the 2AFC file of the data repository, or the folder results/2AFC/formatted
# written by 1_preprocess_pavlovia.py
INPUT = Path(r'data/05_human_similarity_2AFC/2AFC_trials_anonymised.csv')
output_dir = Path(r'results/2AFC/sigmoid_fits')
output_dir.mkdir(parents=True, exist_ok=True)

custom_bins  = [0.000, 0.300, 0.400, 0.457, 0.514, 0.571, 0.629, 0.686, 0.743, 0.800, 1.000]
bin_labels   = [f"{custom_bins[i]:.2f}-{custom_bins[i+1]:.2f}" for i in range(len(custom_bins) - 1)]
bin_centers  = np.array([(custom_bins[i] + custom_bins[i+1]) / 2 for i in range(len(custom_bins) - 1)])

def sigmoid(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def process_file(df: pd.DataFrame, name: str):
    """Fit 75% threshold for a single CSV and save a plot. Returns a dict of results."""
    df = df.dropna(subset=['CLIP Cosine Similarity', 'Response_Binary'])

    # Bin trials
    df['bin'] = pd.cut(
        df['CLIP Cosine Similarity'],
        bins=custom_bins,
        labels=False,
        include_lowest=True
    )

    mean_similar, se_similar, counts_bin = [], [], []

    for i in range(len(bin_centers)):
        group = df[df['bin'] == i]
        if len(group) > 0:
            y = group['Response_Binary'].to_numpy()
            p = y.mean()
            n = y.size
            mean_similar.append(p)
            se_similar.append(np.sqrt(p * (1 - p) / n) if n > 1 else np.nan)
            counts_bin.append(n)
        else:
            mean_similar.append(np.nan)
            se_similar.append(np.nan)
            counts_bin.append(0)

    mean_similar = np.array(mean_similar, dtype=float)
    se_similar   = np.array(se_similar,   dtype=float)
    counts_bin   = np.array(counts_bin,   dtype=int)

    mask   = ~np.isnan(mean_similar)
    x_bin  = bin_centers[mask]
    y_bin  = mean_similar[mask]
    y_err  = se_similar[mask]
    n_bin  = counts_bin[mask]

    result = {
        'participant': name,
        'n_trials':    int(df.shape[0]),
        'n_bins_used': int(mask.sum()),
        'x0':          np.nan,
        'k':           np.nan,
        'threshold_75':np.nan,
        'fit_ok':      False
    }

    if len(x_bin) < 3:
        print(f"[skip] {name}: not enough bins with data.")
        return result

    try:
        p0     = [np.median(x_bin), 10.0]
        bounds = ([x_bin.min(), 1e-3], [x_bin.max(), 1e3])
        popt, _ = curve_fit(sigmoid, x_bin, y_bin, p0=p0, bounds=bounds, maxfev=20000)

        x0, k   = popt
        thr_75  = x0 + np.log(0.75 / 0.25) / k

        result.update({
            'x0':          float(x0),
            'k':           float(k),
            'threshold_75':float(thr_75),
            'fit_ok':      True
        })

        # plot
        fig, ax = plt.subplots(figsize=(5, 4))

        # data points (Vermillion, colorblind-safe)
        ax.errorbar(
            x_bin, y_bin, yerr=y_err,
            fmt='o',
            color=COLOR_DATA,
            capsize=5,
            elinewidth=1.8,
            markersize=7,
            label='% Similar ± SE'
        )

        # trial counts above each point
        for xb, yb, ye, c in zip(x_bin, y_bin, y_err, n_bin):
            y_text = (yb + ye + 0.02) if not np.isnan(ye) else (yb + 0.02)
            ax.text(
                xb, min(y_text, 1.03),
                str(c),
                ha='center', va='bottom',
                fontsize=12,
                color='#444444'
            )

        # sigmoid curve (Blue, colorblind-safe)
        x_vals = np.linspace(custom_bins[0], custom_bins[-1], 400)
        ax.plot(
            x_vals, sigmoid(x_vals, *popt),
            '-', color=COLOR_CURVE, linewidth=2.2,
            label='Sigmoid fit'
        )

        # threshold line (Orange, colorblind-safe)
        ax.axvline(
            thr_75,
            color=COLOR_THRESHOLD,
            linestyle='--',
            linewidth=2.0,
            label=f'75% threshold = {thr_75:.3f}'
        )

        # x-axis
        x_tick_values = np.arange(0.0, 1.01, 0.10)
        ax.set_xticks(x_tick_values)
        ax.set_xticklabels([f"{v:.1f}" for v in x_tick_values], rotation=0)

        # y-axis
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

        ax.set_ylim(-0.05, 1.10)
        ax.set_xlim(0.0, 1.05)

        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#AAAAAA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper left', framealpha=0.75,
                  edgecolor='#CCCCCC', handlelength=1.5)

        plt.tight_layout()

        out_png = output_dir / f"{name}_sigmoid75.png"
        fig.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    except Exception as e:
        print(f"[fit error] {name}: {e}")

    return result


# ============================================================
# RUN ON ALL CSVs
# ============================================================
results = []
COLS = {'clip_cosine_similarity': 'CLIP Cosine Similarity', 'response_similar': 'Response_Binary'}
if INPUT.is_dir():
    for csv_path in sorted(INPUT.glob('*.csv')):
        results.append(process_file(pd.read_csv(csv_path), csv_path.stem))
else:
    trials = pd.read_csv(INPUT).rename(columns=COLS)
    for name, d in trials.groupby('participant'):
        results.append(process_file(d.copy(), name))

summary_df   = pd.DataFrame(results)
summary_path = output_dir / 'sigmoid75_thresholds_summary.csv'
summary_df.to_csv(summary_path, index=False)

print(f"\nSaved summary -> {summary_path}")
print(summary_df[['participant', 'fit_ok', 'threshold_75', 'x0', 'k', 'n_trials', 'n_bins_used']])