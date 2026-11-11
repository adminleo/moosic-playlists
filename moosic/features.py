"""Turning audio features into a space where distance means something."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .data import FEATURES


def matrix(d: pd.DataFrame, how: str = "quantile") -> np.ndarray:
    """Scale the features.

    K-Means minimises squared Euclidean distance, so a feature's influence is
    its variance. In the raw file `duration_ms` has a standard deviation of
    135,633 and `danceability` has 0.218 — a ratio of about 600,000 to one.
    Unscaled, the algorithm is a duration sorter wearing a playlist costume.

    `standard` centres and divides by the standard deviation. `quantile` maps
    each feature onto a normal distribution by rank, which also tames the
    skewed ones (`speechiness` and `instrumentalness` are both piled up near
    zero with a long tail). The practical difference is cluster balance, and
    for this use case that is the deciding criterion — see FINDINGS.
    """
    X = d[FEATURES].to_numpy(dtype=float)
    if how == "standard":
        return StandardScaler().fit_transform(X)
    if how == "quantile":
        return QuantileTransformer(output_distribution="normal",
                                   random_state=0).fit_transform(X)
    if how == "none":
        return X
    raise ValueError(f"unknown scaling: {how}")


def pca_report(d: pd.DataFrame, how: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variance explained, and what the leading component is actually made of.

    Run under both scalings, because the answer differs and the difference is
    the point: PC1 reports the preprocessing choice at least as loudly as it
    reports the music.
    """
    from sklearn.decomposition import PCA

    X = matrix(d, how)
    p = PCA().fit(X)
    ev = p.explained_variance_ratio_
    variance = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(ev))],
        "explained": ev.round(3),
        "cumulative": ev.cumsum().round(3),
    })
    loadings = pd.DataFrame(p.components_[:3].T, index=FEATURES,
                            columns=["PC1", "PC2", "PC3"]).round(2)
    return variance, loadings.reset_index(names="feature")
