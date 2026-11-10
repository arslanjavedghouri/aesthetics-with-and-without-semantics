# Overall LLM-human alignment, AVA vs MSC. AVA bins are merged in pairs to match the 5 MSC bins.
# Reads the summary_results.csv files from the split-half scripts.

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
AVA_EXCEL   = r"results/combined-human-llm-rating-AVA.xlsx"   # from 06_alignment/0_build_alignment_table_AVA.py
AVA_CSV     = r"results/split_half/AVA/1000-iterations/summary_results.csv"

MSC_EXCEL   = r"results/MSC_LLM_ratings.xlsx"   # from 05_llm_rating/rate_captions_MSC.py
MSC_CSV     = r"results/split_half/MSC/1000-iterations/summary_results.csv"

OUTPUT_DIR  = r"results/cross_dataset"

METRIC_COL = 'is_close_enough_combined_euclidean'
N_BOOT = 10000
N_PER_CLASS = 100  # images per class in the subsampling check
SEED = 42


# ============================================================
# LOADING
# ============================================================
def parse_bin(val):
    if pd.isna(val):
        return None
    if isinstance(val, str) and val.startswith('Bin-'):
        return int(val.replace('Bin-', ''))
    try:
        return int(val)
    except Exception:
        return None


def load_counts(excel_file, csv_file, merge_pairs):
    """
    Return per-class (close, total) arrays of length 5.
    merge_pairs=True  -> AVA: raw bins 0-9 merged into classes 1-5
    merge_pairs=False -> MSC: raw bins 1-5 used as classes 1-5
    """
    bmap = {r['Image ID']: parse_bin(r['Bin'])
            for _, r in pd.read_excel(excel_file).iterrows()}
    df = pd.read_csv(csv_file)
    df['bin'] = df['image_id'].map(bmap)
    df = df[df['bin'].notna()].copy()
    df['bin'] = df['bin'].astype(int)
    df['yi'] = df[METRIC_COL].astype(bool).astype(int)

    raw_bins = sorted(df['bin'].unique())
    print(f"  raw bins found: {raw_bins}")
    if merge_pairs:
        assert raw_bins == list(range(10)), f"Expected AVA bins 0-9, got {raw_bins}"
        df['cls'] = df['bin'] // 2 + 1
    else:
        assert raw_bins == [1, 2, 3, 4, 5], f"Expected MSC bins 1-5, got {raw_bins}"
        df['cls'] = df['bin']

    g = df.groupby('cls')['yi'].agg(['sum', 'count'])
    return g['sum'].reindex(range(1, 6)).values.astype(int),\
        g['count'].reindex(range(1, 6)).values.astype(int)


# ============================================================
# ESTIMATORS
# ============================================================
def two_sided_p(boot_diff, obs_diff):
    p = 2 * np.mean(boot_diff <= 0) if obs_diff >= 0 else 2 * np.mean(boot_diff >= 0)
    return max(min(p, 1.0), 2 / len(boot_diff))


def summarize(name, ava_est, msc_est, boot_ava, boot_msc):
    diff = ava_est - msc_est
    bd = boot_ava - boot_msc
    lo, hi = np.percentile(bd, [2.5, 97.5])
    a_lo, a_hi = np.percentile(boot_ava, [2.5, 97.5])
    m_lo, m_hi = np.percentile(boot_msc, [2.5, 97.5])
    p = two_sided_p(bd, diff)
    return dict(name=name, ava=ava_est, msc=msc_est, diff=diff,
                a_lo=a_lo, a_hi=a_hi, m_lo=m_lo, m_hi=m_hi,
                d_lo=lo, d_hi=hi, p=p, frac_ava_higher=np.mean(bd > 0))


def run_analysis(ava_close, ava_total, msc_close, msc_total, rng,
                 n_boot=N_BOOT, n_per_class=N_PER_CLASS):
    p_ava = ava_close / ava_total
    p_msc = msc_close / msc_total
    w_msc = msc_total / msc_total.sum()  # MSC class mix

    # stratified bootstrap draws (within each class, with replacement)
    b_ava = rng.binomial(ava_total, p_ava, size=(n_boot, 5)) / ava_total
    b_msc = rng.binomial(msc_total, p_msc, size=(n_boot, 5)) / msc_total

    results = {}

    # Unweighted global (each dataset's own composition) - Supplement
    obs_a = ava_close.sum() / ava_total.sum() * 100
    obs_m = msc_close.sum() / msc_total.sum() * 100
    bg_a = (b_ava * ava_total).sum(axis=1) / ava_total.sum() * 100
    bg_m = (b_msc * msc_total).sum(axis=1) / msc_total.sum() * 100
    results['unweighted'] = summarize('Unweighted global (own class mix)',
                                      obs_a, obs_m, bg_a, bg_m)

    # A) class-balanced
    results['balanced'] = summarize('A) Class-balanced (equal weight/class)',
                                    p_ava.mean() * 100, p_msc.mean() * 100,
                                    b_ava.mean(axis=1) * 100,
                                    b_msc.mean(axis=1) * 100)

    # B) standardised to MSC's class mix
    results['std_msc'] = summarize("B) Standardised to MSC's class mix",
                                   (p_ava * w_msc).sum() * 100,
                                   (p_msc * w_msc).sum() * 100,
                                   (b_ava * w_msc).sum(axis=1) * 100,
                                   (b_msc * w_msc).sum(axis=1) * 100)

    # C) subsampling: 100 images per class, without replacement, repeated n_boot times
    # (hypergeometric = exact count of "close" images in a random 100-subset)
    assert (ava_total >= n_per_class).all() and (msc_total >= n_per_class).all(),\
        "A class has fewer images than the subsample size"
    s_ava = rng.hypergeometric(ava_close, ava_total - ava_close,
                               n_per_class, size=(n_boot, 5)) / n_per_class
    s_msc = rng.hypergeometric(msc_close, msc_total - msc_close,
                               n_per_class, size=(n_boot, 5)) / n_per_class
    results['subsample'] = summarize(
        f"C) Subsampling ({n_per_class}/class, {n_boot:,} draws)",
        p_ava.mean() * 100, p_msc.mean() * 100,
        s_ava.mean(axis=1) * 100, s_msc.mean(axis=1) * 100)
    return results


def fmt(r):
    return (f"{r['name']}\n"
            f"    AVA: {r['ava']:.1f}%  (95% CI [{r['a_lo']:.1f}, {r['a_hi']:.1f}])\n"
            f"    MSC: {r['msc']:.1f}%  (95% CI [{r['m_lo']:.1f}, {r['m_hi']:.1f}])\n"
            f"    Difference (AVA - MSC): {r['diff']:+.1f} pp  "
            f"(95% CI [{r['d_lo']:.1f}, {r['d_hi']:.1f}], p = {r['p']:.4f}, "
            f"AVA higher in {r['frac_ava_higher'] * 100:.1f}% of resamples)\n")

def leave_one_class_out(ava_close, ava_total, msc_close, msc_total, rng, n_boot=N_BOOT):
    p_ava = ava_close / ava_total
    p_msc = msc_close / msc_total
    b_ava = rng.binomial(ava_total, p_ava, size=(n_boot, 5)) / ava_total * 100
    b_msc = rng.binomial(msc_total, p_msc, size=(n_boot, 5)) / msc_total * 100
    d = (p_ava - p_msc) * 100
    lines = ["PER-CLASS CONTRIBUTION TO THE CLASS-BALANCED DIFFERENCE (AVA - MSC)",
             f"  per-class difference (pp): {np.round(d, 1)}",
             f"  contribution to mean (pp): {np.round(d / 5, 1)}   (mean = {d.mean():.1f})",
             "", "LEAVE-ONE-CLASS-OUT (class-balanced mean over the remaining 4 classes)"]
    for drop in range(5):
        keep = [i for i in range(5) if i != drop]
        bd = (b_ava[:, keep] - b_msc[:, keep]).mean(axis=1)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        lines.append(f"  without class {drop + 1}: {d[keep].mean():+.1f} pp  "
                     f"(95% CI [{lo:.1f}, {hi:.1f}])")
    lines.append("")
    return lines

# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading AVA ...")
    ava_close, ava_total = load_counts(AVA_EXCEL, AVA_CSV, merge_pairs=True)
    print("Loading MSC ...")
    msc_close, msc_total = load_counts(MSC_EXCEL, MSC_CSV, merge_pairs=False)

    lines = []
    lines.append("PER-CLASS (5 merged classes)")
    lines.append(f"{'Class':<6}{'AVA close/total':>20}{'AVA %':>9}"
                 f"{'MSC close/total':>20}{'MSC %':>9}")
    for k in range(5):
        lines.append(f"{k + 1:<6}{f'{ava_close[k]}/{ava_total[k]}':>20}"
                     f"{ava_close[k] / ava_total[k] * 100:>9.1f}"
                     f"{f'{msc_close[k]}/{msc_total[k]}':>20}"
                     f"{msc_close[k] / msc_total[k] * 100:>9.1f}")
    lines.append("")
    lines.append(f"Class mix (share of images)  AVA: "
                 f"{np.round(ava_total / ava_total.sum() * 100, 1)}")
    lines.append(f"                             MSC: "
                 f"{np.round(msc_total / msc_total.sum() * 100, 1)}")
    lines.append(f"Totals: AVA {ava_total.sum()}, MSC {msc_total.sum()}\n")

    res = run_analysis(ava_close, ava_total, msc_close, msc_total, rng)
    for key in ['unweighted', 'balanced', 'std_msc', 'subsample']:
        lines.append(fmt(res[key]))
    lines += leave_one_class_out(ava_close, ava_total, msc_close, msc_total, rng)
    text = "\n".join(lines)
    print("\n" + "=" * 70)
    print(text)
    out = os.path.join(OUTPUT_DIR, 'global_alignment_5v5.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
