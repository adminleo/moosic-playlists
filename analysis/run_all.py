"""Every measurement behind README.md and findings/FINDINGS.md.

    python3 analysis/run_all.py [path/to/songs.csv]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moosic import data, evaluate, features  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "findings", "results")
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "payload.bin")
K = 8


def emit(name, df, head=None):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"\n### {name}")
    print((df.head(head) if head else df).to_string(index=False))


def main() -> int:
    d = data.load(CSV)
    artists = d["artist"].to_numpy()
    print(f"{len(d):,} tracks | {d['artist'].nunique():,} artists | "
          f"{len(data.FEATURES)} features")

    # --- 1. why scaling decides the outcome -------------------------------
    # Run this one WITH duration_ms, because the demonstration is the reason
    # duration_ms is excluded everywhere else.
    cols = data.FEATURES + ["duration_ms"]
    raw = d[cols].to_numpy(dtype=float)
    lab_raw = KMeans(K, n_init=10, random_state=0).fit_predict(raw)
    df = pd.DataFrame(raw, columns=cols).assign(c=lab_raw)
    share = (df.groupby("c")[cols].mean().var(ddof=0)
             / df[cols].var(ddof=0)).sort_values(ascending=False)
    emit("scaling_unscaled_dominance",
         share.rename("between_cluster_variance_share").reset_index()
         .rename(columns={"index": "feature"}).round(3))
    sizes = np.bincount(lab_raw)
    mins = (pd.Series(raw[:, cols.index("duration_ms")]).groupby(lab_raw).mean()
            / 60000).round(1)
    emit("scaling_unscaled_clusters",
         pd.DataFrame({"cluster": range(len(sizes)), "tracks": sizes,
                       "mean_duration_min": mins.to_numpy()})
         .sort_values("mean_duration_min"))
    sd = d[cols].std()
    print(f"   std(duration_ms)={sd['duration_ms']:,.0f} vs "
          f"std(danceability)={sd['danceability']:.3f} "
          f"-> ratio {sd['duration_ms']/sd['danceability']:,.0f}:1")

    # --- 2. choosing k ----------------------------------------------------
    X = features.matrix(d, "quantile")
    rows = []
    for k in (4, 6, 8, 12, 20, 40):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        rows.append({"k": k,
                     "silhouette": round(float(silhouette_score(
                         X, lab, sample_size=3000, random_state=0)), 3),
                     **{kk: vv for kk, vv in evaluate.stability(X, k, seeds=4).items()
                        if kk != "k"},
                     "largest_cluster": int(np.bincount(lab).max()),
                     **{f"artist_{kk}": vv
                        for kk, vv in evaluate.artist_coherence(lab, artists).items()}})
    emit("choosing_k", pd.DataFrame(rows))

    # --- 3. the chosen model ---------------------------------------------
    labels = KMeans(K, n_init=10, random_state=0).fit_predict(X)
    prof = d.groupby(labels)[data.FEATURES].mean()
    z = ((prof - d[data.FEATURES].mean()) / d[data.FEATURES].std()).round(2)
    z.insert(0, "tracks", np.bincount(labels))
    emit("cluster_profiles_z", z.reset_index(names="cluster"))

    emit("evaluation_same_title",
         pd.DataFrame([evaluate.same_title_agreement(d, labels)]))
    emit("evaluation_retrieval",
         evaluate.retrieval_vs_clusters(X, labels, artists))

    # --- 3b. PCA, under both scalings -------------------------------------
    for how in ("quantile", "standard"):
        var, load = features.pca_report(d, how)
        emit(f"pca_variance_{how}", var)
        emit(f"pca_loadings_{how}", load)

    # --- 4. the playlists themselves --------------------------------------
    # Published as Spotify track id + assigned cluster only. The id is a public
    # identifier and the cluster is this project's output; the track list, the
    # names and the audio features are course-provided material and are not
    # redistributed here. Anyone with the dataset can rejoin on `id`.
    (d.assign(cluster=labels)[["cluster", "id"]]
     .sort_values(["cluster", "id"])
     .to_csv(os.path.join(OUT, "playlist_assignments.csv"), index=False))
    # The readable version, for working locally. Not committed.
    (d.assign(cluster=labels)[["cluster", "name", "artist", "id"]]
     .sort_values(["cluster", "artist"])
     .to_csv(os.path.join(ROOT, "playlists_local.csv"), index=False))
    print(f"\nWrote {len(os.listdir(OUT))} files to findings/results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
