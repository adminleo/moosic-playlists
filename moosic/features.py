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
