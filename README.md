# Moosic — can audio features make playlists?

Moosic sells human-curated playlists, each built around a mood. The curators
cannot keep up with growth, so the brief is to prototype automatic playlists by
clustering Spotify audio features, and to answer two questions the team is
already arguing about:

1. Can these features identify songs that *people* would call similar?
2. Is K-Means the right method, or should we look elsewhere?

Both answers here are measured against something outside the model, because the
usual cluster metrics cannot settle either question. Silhouette says how tight a
group is in feature space; it cannot say whether the group is a playlist.

## The answers

**Yes, the features carry real similarity signal.** Using artist agreement as a
proxy a human can verify — two tracks by the same artist are usually more alike
than two at random — the twenty nearest neighbours of a song are by its own
artist **19.5× more often than chance**. Songs drawn from the same K-Means
cluster manage 3.9×. The signal is there and it is not subtle.

**No, K-Means is the wrong shape for this product.** Three measurements:

- **There is no natural number of playlists.** Silhouette falls monotonically
  from 0.309 at k=4 to 0.123 at k=40, with no peak. A peak would mark a
  structure the data actually has; a monotone decline means the feature space
  is a continuum and k is a business choice dressed as a discovery.
- **The clusters describe how a track was recorded, not what it feels like.**
  The separating features are loudness, energy, acousticness and
  instrumentalness. One cluster is quiet acoustic instrumentals (loudness
  −2.02σ, energy −1.85σ, acousticness +1.80σ); another is energetic, hard to
  dance to and joyless (energy +0.91σ, danceability −1.14σ, valence −0.83σ).
  Those are recording and production descriptions. A cluster defined by "sounds
  like it was played live" spans every genre there is.
- **PCA finds no shortcut either.** Reaching 80 % of the variance takes four of
  the nine components. And the leading component is `instrumentalness` with a
  loading of 0.95 under the chosen scaling, but `loudness` and `energy` under
  `StandardScaler` — PC1 reports the preprocessing decision at least as loudly
  as it reports the music, which is why it is run under both.
- **The same song in two recordings usually splits.** 92 titles in this
  catalogue were recorded by more than one artist. Only **22 of them (23.9 %)**
  land in the same cluster. The model is grouping production, not music.

**What to do instead.** Moosic's product is a playlist built around a mood, which
is a *seed and neighbours* problem. Partitioning the whole catalogue is a
different task that merely produces groups as a side effect — and it performs
five times worse on the same yardstick. Nearest-neighbour retrieval from a seed
track is simpler, needs no choice of k, lets a curator steer by picking the seed,
and scores 3.86 % against the cluster's 0.78 %.

Clustering still earns a place as a *coarse first pass* — eight buckets a curator
can work through — but not as the thing that makes the playlist.

## Two traps worth the detour

**Forget to scale and you build a duration sorter.** K-Means minimises squared
Euclidean distance, so a feature's influence is its spread, and `duration_ms`
has 625,120 times the standard deviation of `danceability`. Run it raw and the
eight clusters come out as a ladder of mean track lengths — 2.6, 3.6, 4.7, 6.3,
9.4, 16.0, 25.3 and 61.7 minutes — with one cluster holding two tracks that
average an hour. Duration accounts for 67.2 of the between-cluster variance
share against 1.17 for the next feature. Nothing errors. You get eight
playlists, and they are sorted by length.

**Scaling by rank does not equalise what is tied.** The quantile transform gives
better-balanced clusters — with k=8 the largest holds 801 tracks against
`StandardScaler`'s 1,294, and the whole range is 438–801 against 265–1,294 — which matters
when the output is meant to be a playlist. But 24.5 % of tracks have
`instrumentalness` of exactly 0.0, and tied values cannot be spread: they all
map to one point and the rest are pushed into the tails, leaving that one
feature with **2.5× the spread** — and so 2.5× the pull on every centroid —
while the other eight sit within 5 % of each other. `StandardScaler` equalises
exactly and clusters worse. The trade is documented rather than hidden, and
pinned in `tests/test_moosic.py` so swapping the transform cannot quietly change
which feature dominates.

## Running it

```bash
python3 analysis/run_all.py          # every table, written to findings/results/
python3 -m unittest discover -s tests -v
```

The dataset is bootcamp-provided and not redistributed here. Point the script at
the CSV: `python3 analysis/run_all.py /path/to/songs.csv`.

```
moosic/     data.py (load, de-duplicate, choose features)
            features.py (scaling, and what each choice costs)
            evaluate.py (the tests against artist, not against the model)
analysis/   run_all.py — every number in this README
findings/   FINDINGS.md, the result tables, and the cluster assignments
tests/      the decisions pinned
```

Requires pandas, numpy and scikit-learn. `findings/results/playlist_assignments.csv` holds the cluster assigned to each
Spotify track id — the prototype the brief asked for. Only the id and the
cluster are published: the track list and the audio features are
course-provided material. `analysis/run_all.py` also writes a readable
`playlists_local.csv` with names and artists, which stays local.
