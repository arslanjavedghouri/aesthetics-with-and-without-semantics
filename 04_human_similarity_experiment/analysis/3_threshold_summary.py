import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ========= CONFIG =========
csv_path = Path(r'results/2AFC/sigmoid_fits/sigmoid75_thresholds_summary.csv')   # from 2_fit_sigmoid_thresholds.py

out_dir = Path(r'results/2AFC')
out_dir.mkdir(parents=True, exist_ok=True)

# ========= LOAD & STATS =========
df = pd.read_csv(csv_path)
x = df['threshold_75'].dropna().to_numpy().astype(float)
n = x.size

mean   = float(np.mean(x))
median = float(np.median(x))
sd     = float(np.std(x, ddof=1))
sem    = sd / np.sqrt(n)
tcrit  = stats.t.ppf(0.975, df=n - 1)
ci_lo  = mean - tcrit * sem
ci_hi  = mean + tcrit * sem
q1, q3 = np.percentile(x, [25, 75])
iqr    = float(q3 - q1)

# ========= FIGURE =========
fig, ax = plt.subplots(figsize=(4.5, 5.5))

# Box plot
ax.boxplot(
    x,
    vert=True,
    widths=0.38,
    patch_artist=True,
    positions=[1],
    medianprops=dict(color='#333333', linewidth=1.8),
    boxprops=dict(facecolor='#D4E8F7', linewidth=1.0, alpha=0.85),
    whiskerprops=dict(linewidth=1.0, linestyle='--', color='#555555'),
    capprops=dict(linewidth=1.2, color='#555555'),
    flierprops=dict(marker='', linestyle='none'),
)

# Jittered individual points
rng = np.random.default_rng(42)
jitter = rng.normal(0, 0.045, size=n)
ax.scatter(
    np.ones(n) + jitter, x,
    s=30, color='#2E6DA4', alpha=0.75, zorder=4,
    label=f'Individual (n={n})'
)

# Mean line
ax.hlines(mean, 0.70, 1.30, colors='#CC3333', linewidths=2.2,
          linestyles='-', zorder=5, label=f'Mean = {mean:.2f}')

# 95% CI bracket
ax.hlines([ci_lo, ci_hi], 0.70, 1.30, colors='#CC3333',
          linewidths=1.5, linestyles='--', zorder=5,
          label=f'95% CI [{ci_lo:.2f}, {ci_hi:.2f}]')
ax.vlines([0.70, 1.30], ci_lo, ci_hi, colors='#CC3333',
          linewidths=1.0, zorder=5)

# Formatting
ax.set_ylabel('Threshold (CLIP cosine similarity)', fontsize=12)
ax.set_xticks([1])
ax.set_xticklabels([f'All participants\n(n = {n})'], fontsize=11)
ax.set_xlim(0.55, 1.45)
ax.set_ylim(max(0.0, x.min() - 0.06), min(1.0, x.max() + 0.06))
ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='x', length=0)
ax.spines['bottom'].set_visible(False)

# Legend
ax.legend(fontsize=9, loc='upper right', framealpha=0.75,
          edgecolor='#BBBBBB', handlelength=1.5)

# Stats annotation
stats_str = (f'Mean = {mean:.2f}\nSD = {sd:.2f}\n'
             f'Median = {median:.2f}\nIQR = {iqr:.2f}')
ax.text(0.97, 0.03, stats_str, transform=ax.transAxes,
        fontsize=9, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.85))

plt.tight_layout()

out_path = out_dir / 'threshold_distribution.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved -> {out_path}")
print(f"\nn={n}  mean={mean:.4f}  SD={sd:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")