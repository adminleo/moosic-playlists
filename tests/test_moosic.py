"""The modelling decisions, pinned."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moosic import data, evaluate, features  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "payload.bin")


class TestEvaluation(unittest.TestCase):
    def test_coherence_of_a_perfect_clustering_beats_chance(self):
        artists = np.repeat(["a", "b", "c"], 10)
        perfect = np.repeat([0, 1, 2], 10)
        r = evaluate.artist_coherence(perfect, artists, min_tracks=5)
        self.assertEqual(r["observed"], 1.0)
        self.assertGreater(r["lift"], 1.5)

    def test_coherence_of_a_random_clustering_is_about_chance(self):
        rng = np.random.default_rng(0)
        artists = np.repeat([f"a{i}" for i in range(20)], 10)
        r = evaluate.artist_coherence(rng.integers(0, 8, 200), artists, min_tracks=5)
        self.assertAlmostEqual(r["lift"], 1.0, delta=0.25)

    def test_chance_baseline_is_not_one_over_k(self):
        """The reason the baseline is simulated rather than assumed.

        With few tracks per artist, the most common cluster is high by
        construction: 1/k would claim 0.25 at k=4 where the real figure is
        near 0.5, and every lift computed against it would be doubled.
        """
        rng = np.random.default_rng(1)
        artists = np.repeat([f"a{i}" for i in range(40)], 5)
        r = evaluate.artist_coherence(rng.integers(0, 4, 200), artists, min_tracks=5)
        self.assertGreater(r["chance"], 0.4)

    def test_same_title_agreement_detects_a_split(self):
        d = pd.DataFrame({"name": ["X", "X"], "artist": ["a", "b"], "id": ["1", "2"]})
        self.assertEqual(evaluate.same_title_agreement(d, np.array([0, 1]))["pct"], 0.0)
        self.assertEqual(evaluate.same_title_agreement(d, np.array([0, 0]))["pct"], 100.0)


@unittest.skipUnless(os.path.exists(CSV), "course dataset not present")
class TestAgainstTheRealFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = data.load(CSV)

    def test_column_names_are_stripped(self):
        self.assertIn("artist", self.d.columns)
        self.assertTrue(all(c == c.strip() for c in self.d.columns))

    def test_repeated_track_ids_are_gone(self):
        self.assertEqual(int(self.d["id"].duplicated().sum()), 0)

    def test_categorical_pitch_is_not_treated_as_a_distance(self):
        for c in ("key", "mode", "time_signature", "duration_ms"):
            self.assertNotIn(c, data.FEATURES)

    def test_unscaled_duration_dwarfs_everything(self):
        raw = self.d[data.FEATURES + ["duration_ms"]].std()
        self.assertGreater(raw["duration_ms"] / raw["danceability"], 100_000)

    def test_standard_scaling_equalises_influence_exactly(self):
        spread = features.matrix(self.d, "standard").std(axis=0)
        self.assertAlmostEqual(spread.max() / spread.min(), 1.0, places=6)

    def test_quantile_scaling_leaves_instrumentalness_oversized(self):
        """A quantile transform cannot spread values that are tied.

        24.5 % of tracks have instrumentalness exactly 0.0. Those all map to
        one output value, and the remaining three quarters are pushed into the
        tails, so the feature ends up with about 2.5x the spread of the other
        eight — and therefore about 2.5x the pull on a K-Means centroid. Every
        other feature lands within 5 % of 1.0.

        This is the cost of the quantile transform, paid for more balanced
        clusters. It is asserted rather than described so that swapping the
        transform cannot quietly change which features dominate.
        """
        spread = pd.Series(features.matrix(self.d, "quantile").std(axis=0),
                           index=data.FEATURES)
        self.assertGreater(spread["instrumentalness"], 2.0)
        others = spread.drop("instrumentalness")
        self.assertLess(others.max() / others.min(), 1.1)
        self.assertGreater(float((self.d["instrumentalness"] == 0).mean()), 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
