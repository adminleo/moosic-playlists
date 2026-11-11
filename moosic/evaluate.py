"""Does the clustering group songs the way a person would?

The brief asks two questions the usual cluster metrics cannot answer. Silhouette
says how tight the groups are in feature space; it cannot say whether a group is
a playlist. These functions test against something outside the features.

The external signal used here is the **artist**. It is a weak proxy for human
similarity — artists change style, and two artists can sound alike — but it is
independent of every column fed to the model, which is the property that matters.
A measure computed from the same numbers the model optimised cannot falsify it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


def artist_coherence(labels: np.ndarray, artists: np.ndarray,
                     min_tracks: int = 5, trials: int = 30,
                     seed: int = 0) -> dict:
    """Share of an artist's tracks landing in their most common cluster.

    Reported against a chance baseline obtained by shuffling the cluster labels,
    which preserves both the cluster sizes and the artists' track counts. Without
    that baseline the raw number is unreadable: with four clusters, a coin would
    score about 0.55.
    """
    rng = np.random.default_rng(seed)

    def purity(lab):
        t = pd.DataFrame({"artist": artists, "cluster": lab})
        counts = t["artist"].value_counts()
        t = t[t["artist"].isin(counts[counts >= min_tracks].index)]
        return float(t.groupby("artist")["cluster"]
                     .agg(lambda s: s.value_counts().iloc[0] / len(s)).mean())

    observed = purity(labels)
    chance = float(np.mean([purity(rng.permutation(labels)) for _ in range(trials)]))
    counts = pd.Series(artists).value_counts()
    return {"observed": round(observed, 3), "chance": round(chance, 3),
            "lift": round(observed / chance, 2),
            "artists_tested": int((counts >= min_tracks).sum())}


def stability(X: np.ndarray, k: int, seeds: int = 5) -> dict:
    """Does the same k give the same grouping when the start point changes?

    K-Means converges to a local optimum, so an unstable result means the
    partition is an artefact of initialisation rather than a property of the
    data. Adjusted Rand corrects for agreement expected by chance.
    """
    labs = [KMeans(k, n_init=10, random_state=s).fit_predict(X) for s in range(seeds)]
    pairs = [adjusted_rand_score(labs[i], labs[j])
             for i in range(seeds) for j in range(i + 1, seeds)]
    return {"k": k, "ari_min": round(float(np.min(pairs)), 3),
            "ari_median": round(float(np.median(pairs)), 3)}


def same_title_agreement(d: pd.DataFrame, labels: np.ndarray) -> dict:
    """Titles recorded by more than one artist: same cluster, or not?

    The strongest available test of whether a cluster tracks the song or the
    recording. If two performances of one composition are routinely assigned to
    different playlists, the model is grouping production, not music.
    """
    t = d.assign(cluster=labels)
    multi = t[t.duplicated("name", keep=False)].groupby("name").filter(
        lambda g: g["artist"].nunique() > 1)
    if multi.empty:
        return {"titles": 0, "same_cluster": 0, "pct": float("nan")}
    agree = multi.groupby("name")["cluster"].agg(lambda s: s.nunique() == 1)
    return {"titles": int(agree.size), "same_cluster": int(agree.sum()),
            "pct": round(100 * float(agree.mean()), 1)}


def retrieval_vs_clusters(X: np.ndarray, labels: np.ndarray, artists: np.ndarray,
                          n: int = 20, min_tracks: int = 5, seed: int = 0) -> pd.DataFrame:
    """Three ways to suggest n songs, scored by how often the artist matches.

    Moosic sells a playlist with a mood, which is a *seed and neighbours*
    problem. Partitioning the whole catalogue is a different task that happens
    to produce groups. This measures both against the same yardstick.
    """
    rng = np.random.default_rng(seed)
    _, idx = NearestNeighbors(n_neighbors=n + 1).fit(X).kneighbors(X)
    idx = idx[:, 1:]                                   # drop the song itself

    knn = np.array([np.mean(artists[row] == artists[i]) for i, row in enumerate(idx)])
    cluster = np.empty(len(X))
    for i in range(len(X)):
        pool = np.flatnonzero(labels == labels[i])
        pool = pool[pool != i]
        pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
        cluster[i] = np.mean(artists[pick] == artists[i])
    random = np.array([np.mean(artists[rng.choice(len(X), n, replace=False)] == artists[i])
                       for i in range(len(X))])

    counts = pd.Series(artists).value_counts()
    m = pd.Series(artists).map(counts).ge(min_tracks).to_numpy()
    base = random[m].mean()
    return pd.DataFrame({
        "method": [f"{n} random from the catalogue",
                   f"{n} random from the same cluster",
                   f"{n} nearest neighbours"],
        "same_artist_pct": [round(100 * random[m].mean(), 2),
                            round(100 * cluster[m].mean(), 2),
                            round(100 * knn[m].mean(), 2)],
        "times_chance": [1.0, round(cluster[m].mean() / base, 1),
                         round(knn[m].mean() / base, 1)],
    })
