# QIP features -> aesthetic class (5 or 3 classes) with an RBF-SVM, for AVA and MSC.
# 10 x repeated stratified 5-fold CV (80/20); imputation and scaling are fitted on the training folds only.
# AVA vs MSC: corrected resampled t-test (Nadeau & Bengio, 2003).

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
N_CLASSES = 5          # 5 (main analysis, Figure 4) or 3 (robustness, Supplementary Figure S3)
N_SPLITS  = 5              # 5-fold -> each split is 80% train / 20% test
N_REPEATS = 10             # 10 repeats x 5 folds = 50 train/test splits per dataset
SEED      = 0

# from 3_class_labels_MSC.py and 4_match_AVA_QIP_to_LLM_set.py, or the QIP tables of the data repository
# (data/07_QIP_features_and_classifier/MSC_QIP_features.csv, AVA_QIP_features.csv)
MSC_CSV = r"results/QIP/MSC_QIP_with_class.csv"
AVA_CSV = r"results/QIP/AVA_QIP_matched_to_LLM_set.csv"
OUT_DIR = Path(r"results/classifier")

FONT_SIZE = 22
CM_MODE = "mean_fold"          # "mean_fold": average number of test images per cell (one test fold, ~20%)
                               # "percent"  : row-normalised %, each row sums to 100 (diagonal = class recall)
SHOW_ERROR_BARS = True         # mean +/- SD across the 50 splits
LETTERS = {"MSC": ("c", "d"),  # (class-wise accuracy panel, confusion-matrix panel)
           "AVA": ("a", "b")}

plt.rcParams.update({
    "font.size":          FONT_SIZE,
    "axes.titlesize":     FONT_SIZE,
    "axes.labelsize":     FONT_SIZE,
    "xtick.labelsize":    FONT_SIZE,
    "ytick.labelsize":    FONT_SIZE,
    "legend.fontsize":    FONT_SIZE,
    "figure.titlesize":   FONT_SIZE
})


# ============================================================
# DATA
# ============================================================
def load_dataset(path):
    df = pd.read_csv(path).rename(columns={"image_file": "img_file", "class_5": "Class Name"})
    y = df["Class Name"].astype(int).values
    if N_CLASSES == 3:
        y = np.where(np.isin(y, [1, 2]), 1, np.where(y == 3, 2, 3))
    X = (df.drop(columns=["img_file", "Class Name"], errors="ignore")
           .select_dtypes(include=[np.number]).values.astype(float))
    return X, y


# ============================================================
# CROSS-VALIDATION
# ============================================================
def fit_predict(X, y, train_idx, test_idx):
    model = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),   # fitted on training data only
        ("scale",  StandardScaler()),                 # fitted on training data only
        ("svm",    SVC(class_weight="balanced")),
    ])
    model.fit(X[train_idx], y[train_idx])
    return test_idx, model.predict(X[test_idx])


def run_cv(X, y, classes):
    splits = list(RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X, y))
    out = Parallel(n_jobs=-1)(delayed(fit_predict)(X, y, tr, te) for tr, te in splits)

    acc, bal, rec = [], [], []
    cm_sum = np.zeros((len(classes), len(classes)))
    for te, pred in out:
        cm = confusion_matrix(y[te], pred, labels=classes)
        cm_sum += cm
        recall = np.diag(cm) / cm.sum(axis=1)
        acc.append(np.trace(cm) / cm.sum())
        bal.append(recall.mean())
        rec.append(recall)
    return dict(acc=np.array(acc), bal=np.array(bal), rec=np.array(rec),
                # AVERAGE test-fold confusion matrix: mean over all N_SPLITS x N_REPEATS
                # (= 50) test folds, so it counts ~20% of the images (one test fold).
                cm=cm_sum / (N_SPLITS * N_REPEATS))


def holdout_confusion(X, y, classes):
    """Single stratified 80/20 split (fixed seed)."""
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    _, pred = fit_predict(X, y, tr, te)
    return confusion_matrix(y[te], pred, labels=classes)


def corrected_comparison(a, b):
    """Nadeau-Bengio corrected resampled t-test for the difference of two means
    of repeated-CV scores a, b (independent datasets). Returns diff, CI, p."""
    J = len(a)
    ratio = 1.0 / (N_SPLITS - 1)                    # n_test / n_train
    var_a = (1.0 / J + ratio) * a.var(ddof=1)
    var_b = (1.0 / J + ratio) * b.var(ddof=1)
    se = np.sqrt(var_a + var_b)
    d = a.mean() - b.mean()
    df = J - 1
    t = d / se
    p = 2 * stats.t.sf(abs(t), df)
    half = stats.t.ppf(0.975, df) * se
    return d, d - half, d + half, p


# ============================================================
# FIGURES
# ============================================================
def save_fig(out_stem):
    plt.tight_layout()
    plt.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_stem}.svg", bbox_inches="tight")
    plt.close()


def plot_class_accuracy(letter, name, classes, acc, sd, out_stem, with_bars=None):
    if with_bars is None:
        with_bars = SHOW_ERROR_BARS
    plt.figure(figsize=(8, 6))
    bars = sns.barplot(x=np.asarray(classes).astype(str), y=acc, color="#a11e30ff")
    if with_bars:
        plt.errorbar(np.arange(len(classes)), acc, yerr=sd, fmt="none", color="#333333",
                     capsize=4, capthick=1.3, elinewidth=1.3)
    for bar, a, e in zip(bars.patches, acc, sd):
        height = bar.get_height() + (e if with_bars else 0)
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{a:.2f}",
                 ha='center', va='bottom', fontsize=FONT_SIZE)
    plt.xlabel("Class", fontsize=FONT_SIZE)
    plt.ylabel("Accuracy", fontsize=FONT_SIZE)
    plt.title(rf"$\mathbf{{({letter})}}$ {name} Class-wise Accuracy", fontsize=FONT_SIZE)
    plt.ylim(0, 1.1)
    save_fig(out_stem)


def plot_confusion(letter, name, classes, cm, out_stem):
    plt.figure(figsize=(8, 6))
    if CM_MODE == "percent":
        data = 100.0 * cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(data, annot=True, fmt='.0f', vmin=0, vmax=100,
                    xticklabels=classes, yticklabels=classes, cmap='Reds',
                    cbar_kws={"label": "% of images in true class"})
    else:
        sns.heatmap(np.rint(cm).astype(int), annot=True, fmt='d',
                    xticklabels=classes, yticklabels=classes, cmap='Reds')
    plt.xlabel("Predicted aesthetic class", fontsize=FONT_SIZE)
    plt.ylabel("Human-derived aesthetic class", fontsize=FONT_SIZE)
    plt.title(rf"$\mathbf{{({letter})}}$ {name} Confusion Matrix", fontsize=FONT_SIZE)
    save_fig(out_stem)


# ============================================================
# MAIN
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, path in [("MSC", MSC_CSV), ("AVA", AVA_CSV)]:
        print(f"\n=== {name}: {N_CLASSES}-class ===")
        X, y = load_dataset(path)
        classes = np.unique(y)
        print(f"  images: {len(y)}   features: {X.shape[1]}   "
              f"class counts: {[(y == k).sum() for k in classes]}")
        res[name] = run_cv(X, y, classes)
        la, lc = LETTERS[name]
        rec = res[name]["rec"]
        # (1) cross-validated figures: class-wise accuracy = mean over the 50 splits,
        # confusion matrix = average over all 50 test folds (CM_MODE sets counts or %)
        plot_class_accuracy(la, name, classes, rec.mean(axis=0), rec.std(axis=0, ddof=1),
                            str(OUT_DIR / f"{name}-{N_CLASSES}classes_class_accuracy_cv"))
        plot_confusion(lc, name, classes, res[name]["cm"],
                       str(OUT_DIR / f"{name}-{N_CLASSES}classes_confusion_matrix_pooled_cv"))
        # (2) figures for a single 80/20 hold-out split (test set = 20% of the images)
        cm_h = holdout_confusion(X, y, classes)
        res[name]["hold"] = cm_h
        plot_class_accuracy(la, name, classes, np.diag(cm_h) / cm_h.sum(axis=1),
                            np.zeros(len(classes)),
                            str(OUT_DIR / f"{name}-{N_CLASSES}classes_class_accuracy_holdout"),
                            with_bars=False)
        plot_confusion(lc, name, classes, cm_h,
                       str(OUT_DIR / f"{name}-{N_CLASSES}classes_confusion_matrix_holdout"))

    J = N_SPLITS * N_REPEATS
    lines = [f"QIP + SVM, {N_CLASSES}-class | {N_REPEATS}x{N_SPLITS}-fold stratified CV "
             f"({J} splits, 80/20) | seed={SEED}\n"]
    for key, label in [("acc", "Overall accuracy"), ("bal", "Balanced accuracy")]:
        lines.append(label)
        for name in ("MSC", "AVA"):
            v = res[name][key]
            lines.append(f"  {name}: mean {v.mean():.3f} +/- SD {v.std(ddof=1):.3f}")
        d, lo, hi, p = corrected_comparison(res["MSC"][key], res["AVA"][key])
        lines.append(f"  Difference (MSC - AVA): {d:+.3f}  (95% CI [{lo:.3f}, {hi:.3f}], "
                     f"corrected resampled t-test, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})\n")

    lines.append("Average test fold (confusion matrix cells sum to this; should be ~20% of the images)")
    for name in ("MSC", "AVA"):
        lines.append(f"  {name}: {res[name]['cm'].sum():.0f} images per test fold")
    lines.append("")

    lines.append("Single 80/20 hold-out split (seed fixed at SEED) - only for the alternative old-style figures")
    for name in ("MSC", "AVA"):
        cmh = res[name]["hold"]
        lines.append(f"  {name}: overall accuracy {np.trace(cmh) / cmh.sum():.3f}, balanced accuracy "
                     f"{np.mean(np.diag(cmh) / cmh.sum(axis=1)):.3f}  (test images: {int(cmh.sum())})")
    lines.append("")

    text = "\n".join(lines)
    print("\n" + "=" * 70 + "\n" + text)
    (OUT_DIR / f"summary_{N_CLASSES}class.txt").write_text(text, encoding="utf-8")
    print(f"Saved results and figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()