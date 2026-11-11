"""Load and clean the Spotify audio-feature export."""
from __future__ import annotations

import pandas as pd

# The nine continuous audio features. Deliberately excluded:
#
#   key, mode, time_signature  — categorical. `key` runs 0-11 as pitch classes,
#       so Euclidean distance would call B (11) maximally far from C (0) when
#       they are a semitone apart and enharmonically adjacent. Encoding a
#       circle as a line is a modelling error, not a scaling one.
#   duration_ms  — see `analysis/run_all.py`: it is the single feature that
#       unscaled K-Means clusters on, and even after scaling it still claims a
#       cluster of its own (46 tracks at +7.3 sigma). A playlist of long songs
#       is not a mood.
#   type, html, Unnamed: 0 — empty, a URL, and a row number.
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

TEXT = ["name", "artist", "id"]


def load(path: str) -> pd.DataFrame:
    """Read the export, strip the padding, drop repeated tracks.

    Every column name in the file is padded with trailing spaces — `'artist'`
    is stored as 64 characters. Without the strip, every reference to a column
    needs the exact padding, which is how this dataset produces KeyErrors that
    look like typos.
    """
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    for c in TEXT:
        d[c] = d[c].astype("string").str.strip()
    # 64 rows repeat a Spotify track id. A duplicated track is one track that
    # votes twice: it pulls a centroid toward itself and inflates any measure
    # of how well the clustering groups an artist.
    return d.drop_duplicates("id").reset_index(drop=True)
