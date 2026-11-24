# Five minutes on Moosic

The slides, as text. For presenting, open [`slides.html`](slides.html) and use
the arrow keys. Every number here comes from a table in
[`../findings/results/`](../findings/results/).

---

## 1 · The question

Moosic groups songs into playlists by hand. It does not scale.

Spotify publishes nine numeric features per track — danceability, energy,
loudness, speechiness, acousticness, instrumentalness, liveness, valence,
tempo. **Can K-Means turn those into playlists?**

The honest version of that question has two halves, and they have different
answers: *is there similarity signal in the features at all*, and *is
partitioning the catalogue the right way to use it*.

---

## 2 · Yes — the signal is real

To check it without asking anyone's opinion, use a proxy a human can verify:
two tracks by the same artist are usually more alike than two at random.

| Twenty songs drawn… | share by the same artist | versus chance |
|---|---:|---:|
| at random from the catalogue | 0.20 % | 1× |
| at random from the same cluster | 0.78 % | **3.9×** |
| as the nearest neighbours of a seed | 3.86 % | **19.5×** |

The features carry real structure, and it is not subtle. But the seed already
beats the cluster **five times over**, which is the first hint about what the
product should actually do.

---

## 3 · No — the clusters describe the recording, not the feeling

Three measurements, each of which alone would be arguable, and together are not.

**There is no natural number of playlists.** Silhouette falls from 0.309 at
k=4 to 0.123 at k=40 without a peak. A peak marks structure the data has; a
monotone decline means the feature space is a continuum and *k* is a business
decision wearing a lab coat.

**The separating features are production, not mood.** One cluster is quiet
acoustic instrumentals (loudness −2.02σ, energy −1.85σ, acousticness +1.80σ).
Another is energetic, hard to dance to and joyless (energy +0.91σ,
danceability −1.14σ, valence −0.83σ). "Sounds like it was played live" spans
every genre there is.

**The same song, recorded twice, usually splits.** 92 titles in this catalogue
exist in two different artists' recordings. Only **22 of them — 23.9 % — land
in the same cluster.** A model grouping music would keep them together. This
one is grouping how they were recorded.

---

## 4 · The trap: forget to scale, get a ladder

Run K-Means on the raw features and it returns eight clusters that look
perfectly respectable. They are sorted by track length.

| Cluster | Tracks | Mean duration |
|---|---:|---:|
| 6 | 1,228 | 2.6 min |
| 0 | 2,121 | 3.6 min |
| 4 | 1,223 | 4.7 min |
| 2 | 450 | 6.3 min |
| 7 | 100 | 9.4 min |
| 1 | 35 | 16.0 min |
| 5 | 12 | 25.3 min |
| 3 | 2 | 61.7 min |

Tempo runs to ~200 and duration to millions of milliseconds, so the distance
metric is effectively one feature. **No error is raised.** The output is a
plausible-looking list of playlists that sorts songs by how long they are.

This is the whole argument for checking what a metric measures before
reporting it.

---

## 5 · What to do instead

Moosic's product is a playlist built around a mood — that is **seed and
neighbours**, not partitioning.

- No choice of *k* to defend.
- A curator steers by picking the seed.
- It scores **3.86 % against the cluster's 0.78 %** on the same yardstick.

Clustering keeps a job: **a coarse first pass**, eight buckets a curator can
work through. It just is not the thing that makes the playlist.

> One sentence to take away: the features were good enough, and the method was
> wrong — and no accuracy number anywhere would have told us, because this
> failure only shows up when you ask what the clusters are made of.
