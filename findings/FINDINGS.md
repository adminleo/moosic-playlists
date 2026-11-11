# Findings

Every figure comes from `analysis/run_all.py` and is written to
`findings/results/`. 5,171 tracks by 2,176 artists after de-duplication.

## 0. Cleaning

- **Column names are padded.** Every header in the export carries trailing
  spaces (`'artist'` is stored in a 64-character field). Stripped on load;
  without it, each column reference needs the exact padding and the resulting
  `KeyError` looks like a typo.
- **64 repeated Spotify track ids** removed. A duplicated track votes twice: it
  pulls a centroid toward itself and inflates any measure of how well the
  clustering groups an artist.
- **`key`, `mode` and `time_signature` are excluded.** They are categorical.
  `key` encodes pitch classes 0–11, so Euclidean distance treats B (11) as
  maximally distant from C (0) when they are a semitone apart. Encoding a
  circle as a line is a modelling error, not a scaling one.
- **`duration_ms` is excluded.** See §1.

## 1. Scaling decides the outcome
`scaling_unscaled_dominance.csv`, `scaling_unscaled_clusters.csv`

K-Means minimises squared Euclidean distance, so a feature's influence is its
spread. In the raw file:

    std(duration_ms) = 136,339      std(danceability) = 0.218
    ratio 625,120 : 1

Run K-Means unscaled on all ten continuous features and `duration_ms` accounts
for a between-cluster variance share of **67.2**, against **1.17** for loudness,
the next in line. The eight clusters come out as a ladder of track length:

| mean duration | tracks |
|---:|---:|
| 2.6 min | 1,228 |
| 3.6 min | 2,121 |
| 4.7 min | 1,223 |
| 6.3 min | 450 |
| 9.4 min | 100 |
| 16.0 min | 35 |
| 25.3 min | 12 |
| **61.7 min** | **2** |

Nothing raises an error. You get eight playlists and they are sorted by length,
including one holding two tracks that average an hour. Duration is dropped
entirely rather than scaled, because after scaling it still claimed a cluster
of its own (46 tracks at +7.3σ) and a playlist of long songs is not a mood.

## 2. A quantile transform cannot spread tied values
`tests/test_moosic.py`

`StandardScaler` equalises the nine features exactly — every standard deviation
is 1.0. The quantile transform does not:

| feature | spread after quantile transform |
|---|---:|
| speechiness … valence | 1.002 – 1.004 |
| acousticness | 1.047 |
| **instrumentalness** | **2.498** |

**24.5 % of tracks have `instrumentalness` of exactly 0.0.** A rank-based
transform maps every tied value to one output, so a quarter of the data
collapses onto a point and the remaining three quarters are pushed into the
tails. That one feature ends up with two and a half times the spread, and so two
and a half times the pull on every centroid.

The quantile transform is still chosen, because the product is playlists and it
balances them far better: at k=8 its clusters run 438–801 tracks against
`StandardScaler`'s 265–1,294. The cost is stated and asserted in the tests
rather than left to be discovered.

## 3. There is no natural number of playlists
`choosing_k.csv`

| k | silhouette | ARI (median, 4 seeds) | largest cluster | artist lift |
|---:|---:|---:|---:|---:|
| 4 | 0.309 | 1.000 | 1,760 | 1.75× |
| 6 | 0.197 | 0.952 | 1,181 | 1.99× |
| 8 | 0.168 | 0.885 | 801 | 1.94× |
| 12 | 0.154 | 0.740 | 658 | 1.84× |
| 20 | 0.139 | 0.752 | 363 | 1.93× |
| 40 | 0.123 | 0.522 | 285 | 1.94× |

Silhouette declines monotonically with no peak. A peak marks a structure the
data has; a monotone decline means the space is a continuum and the algorithm is
slicing it. k is therefore a business decision — how many playlists do we want
and how long may they be — not a discovery. k=8 is chosen because it is the
largest k whose clusters remain stable (ARI 0.885) and whose largest playlist is
still a plausible length.

Stability is worth separating from quality. At k=4 the partition is perfectly
reproducible across seeds (ARI 1.000) and still almost structureless
(silhouette 0.309). Reproducible and meaningful are different claims.

## 4. The features do carry human-detectable similarity
`evaluation_retrieval.csv`

Artist agreement is the external yardstick: independent of every column the
model sees, and something a person can check by listening.

| 20 suggested tracks, drawn from | by the same artist | vs chance |
|---|---:|---:|
| the whole catalogue | 0.20 % | 1× |
| the same K-Means cluster | 0.78 % | **3.9×** |
| the 20 nearest neighbours | 3.86 % | **19.5×** |

So the answer to the team's first question is yes — the audio features encode
real similarity, strongly. The sceptics are wrong about that.

They are right about the second part. Cluster membership captures a fifth of
what simple nearest-neighbour retrieval captures from the identical feature
space. The information is in the features; partitioning throws most of it away.

## 5. The clusters group production, not music
`cluster_profiles_z.csv`, `evaluation_same_title.csv`

Cluster profiles in standard deviations from the catalogue mean:

| cluster | tracks | what sets it apart |
|---:|---:|---|
| 6 | 655 | loudness −2.02, energy −1.85, acousticness +1.80, instrumentalness +1.28 |
| 2 | 726 | danceability −1.14, energy +0.91, instrumentalness +0.86, valence −0.83 |
| 4 | 589 | danceability +1.12, valence +1.01 |
| 7 | 438 | danceability −1.01, energy +0.94, acousticness −0.81, valence −0.83 |
| 1 | 676 | danceability +0.85, speechiness +0.90, valence +0.70 |

Every one of those is a description of a recording: how loud, how dense, how
acoustic, whether anyone is singing. None is a genre and none is a mood.

The sharpest test available: **92 titles in this catalogue were recorded by more
than one artist. Only 22 of them — 23.9 % — land in the same cluster.** Three
times out of four the same composition is split across playlists because the two
performances were produced differently. Whatever the clusters are tracking, it is
not the song.

## 6. Recommendation

Keep the feature space; change the algorithm.

- **Nearest-neighbour retrieval from a seed track** matches what Moosic sells —
  a playlist around a mood — needs no choice of k, lets a curator steer by
  choosing the seed, and scores five times better on the only external measure
  available here.
- **Keep clustering as a coarse first pass.** Eight buckets are a reasonable way
  to hand a curator a starting shelf, and `findings/results/playlists.csv` is
  exactly that prototype. It is not the finished product.
- **Collect a human label before the next iteration.** Every number in §4 and §5
  leans on artist agreement because nothing better exists in this dataset. A few
  thousand curator judgements — "do these two belong on one playlist?" — would
  turn every question here from a proxy into a measurement, and would make it
  possible to train toward what Moosic actually sells instead of toward
  whatever the audio features happen to separate.
