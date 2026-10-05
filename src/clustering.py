"""Unsupervised analysis: PCA, K-Means, DBSCAN.

Labels are NOT used to build clusters; they are only used afterwards to
measure agreement (Adjusted Rand Index) with the high/low risk outcome.
Features are scaled first (otherwise 'chol' and 'thalachh' dominate distances).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from common import RESULTS, TARGET, RANDOM_STATE, get_xy, load_data, make_preprocessor


def main():
    df = load_data()
    X, y = get_xy(df)
    Z = make_preprocessor().fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z

    # PCA variance
    pca_full = PCA().fit(Z)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    print("Cumulative explained variance (first 5 PCs):", cum[:5].round(3))

    # K-Means: tune number of PCA components, k fixed to 2 (binary outcome)
    rows = []
    for n in range(2, 11):
        P = PCA(n_components=n, random_state=RANDOM_STATE).fit_transform(Z)
        lab = KMeans(2, n_init=10, random_state=RANDOM_STATE).fit_predict(P)
        rows.append(dict(method="kmeans", param=f"pcs={n}",
                         ari=adjusted_rand_score(y, lab),
                         silhouette=silhouette_score(P, lab)))
    # DBSCAN: tune eps (on scaled data) and min_samples
    P2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Z)
    best_db = None
    for eps in np.arange(0.5, 6.01, 0.25):
        for ms in (3, 5, 10):
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
            ncl = len(set(lab)) - (1 if -1 in lab else 0)
            ari = adjusted_rand_score(y, lab)
            sil = silhouette_score(Z, lab) if ncl > 1 else np.nan
            rows.append(dict(method="dbscan", param=f"eps={eps:.2f},min={ms}",
                             ari=ari, silhouette=sil, clusters=ncl,
                             noise=int((lab == -1).sum())))
            if best_db is None or ari > best_db[0]:
                best_db = (ari, eps, ms, lab)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "clustering_results.csv", index=False)
    km = res[res.method == "kmeans"].sort_values("ari", ascending=False).iloc[0]
    print("Best K-Means:", km.param, f"ARI={km.ari:.3f}")
    print("Best DBSCAN:", f"eps={best_db[1]:.2f}, min_samples={best_db[2]}",
          f"ARI={best_db[0]:.3f}")

    n_best = int(km.param.split("=")[1])
    Pk = PCA(n_components=n_best, random_state=RANDOM_STATE).fit_transform(Z)
    km_lab = KMeans(2, n_init=10, random_state=RANDOM_STATE).fit_predict(Pk)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, lab, t in zip(ax, [y.values, km_lab, best_db[3]],
                         ["True outcome", f"K-Means (ARI={km.ari:.2f})",
                          f"DBSCAN (ARI={best_db[0]:.2f})"]):
        a.scatter(P2[:, 0], P2[:, 1], c=lab, cmap="coolwarm", s=18, alpha=.8)
        a.set_title(t); a.set_xlabel("PC1"); a.set_ylabel("PC2")
    plt.tight_layout(); plt.savefig(RESULTS / "clustering.png", dpi=150); plt.close()

    # Phenotype profile of K-Means clusters (what do the clusters look like?)
    prof = X.assign(cluster=km_lab, outcome=y.values).groupby("cluster").mean().T
    prof.to_csv(RESULTS / "kmeans_cluster_profiles.csv")
    print(prof.round(2))


if __name__ == "__main__":
    main()
