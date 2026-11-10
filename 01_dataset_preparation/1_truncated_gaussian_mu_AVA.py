# Truncated Gaussian fit of each AVA rating histogram (10 rating values, scale 1-10).
# The fitted mean (MU) gives the aesthetic bin used in the paper: bin = int(MU) + 1  (bins 1-10).
# compute_mu() follows Fit_trunc_gaussian.m (MATLAB).
import numpy as np
import pandas as pd
from scipy.optimize import fmin
from scipy.stats import norm
from tqdm import tqdm

AVA_FILE = "data/AVA/AVA.txt"                  # from the AVA dataset (Murray et al., 2012)
OUT_FILE = "results/AVA_ratings_with_MU.xlsx"   # input for the next scripts

# columns of AVA.txt: index, image ID, number of ratings for each score 1-10, two semantic tag IDs, challenge ID
AVA_COLUMNS = (["Index", "Image ID"] + [f"Rating {i}" for i in range(1, 11)]
               + ["Tag ID 1", "Tag ID 2", "Challenge ID"])


def compute_mu(votes_raw):
    votes  = np.asarray(votes_raw, dtype=float)
    n_bins = len(votes)
    total  = votes.sum()
    if total == 0:
        return np.nan, np.nan

    counts = votes / total
    step   = 1
    minn, maxn = 1, n_bins
    X      = np.arange(minn - maxn, 2 * maxn + step, step, dtype=float)

    def objective(x):
        mu_x, sigma_x = float(x[0]), float(x[1])
        if sigma_x <= step / 2:
            return 10.0
        mycdf   = norm.cdf(X - step / 2, loc=mu_x, scale=sigma_x)
        counts2 = np.diff(mycdf)
        ind     = (X[:-1] >= minn) & (X[:-1] <= maxn)
        counts2 = counts2[ind] / counts2[ind].sum()
        rmse    = np.sqrt(np.sum((counts - counts2) ** 2))
        return rmse + sigma_x / 10.0 + 1.0 / (100.0 * sigma_x)

    bins   = np.arange(1, n_bins + 1, dtype=float)
    mu0    = float(np.dot(counts, bins))
    sigma0 = max(float(np.sqrt(np.dot(counts, (bins - mu0) ** 2))), 0.51)

    result = fmin(objective, [mu0, sigma0],
                  xtol=1e-10, ftol=1e-10,
                  maxiter=10_000, maxfun=10_000, disp=False)
    return float(result[0]), float(result[1])


if __name__ == "__main__":
    df = pd.read_csv(AVA_FILE, sep=r"\s+", header=None, names=AVA_COLUMNS)
    cols = [f"Rating {i}" for i in range(1, 11)]
    res = [compute_mu(r[cols].values) for _, r in tqdm(df.iterrows(), total=len(df))]
    df["MU"] = [m for m, _ in res]
    df["Sigma"] = [s for _, s in res]
    df.to_excel(OUT_FILE, index=False)
