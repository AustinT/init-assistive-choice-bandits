import math

from ind_acb.misc import beta_dist_max_prob


class TestBetaDistMaxProb:
    def test_equal(self):
        alpha = [1, 1, 1]
        beta = [1, 1, 1]

        for i in range(3):
            assert math.isclose(beta_dist_max_prob(alpha, beta, i), 1 / 3)

    def test_skewed(self):
        """Two beta distributions, extremely skewed."""
        alpha = [1e-5, 1]
        beta = [1, 2]
        assert 0 < beta_dist_max_prob(alpha, beta, 0) < 1e-4
        assert 1 - 1e-4 < beta_dist_max_prob(alpha, beta, 1) < 1

    def test_hand_picked(self):
        """A hand-picked example which I verified myself using samples."""
        alpha = [5, 1, 1]
        beta = [6, 3, 1]

        expected = [0.38461538461538447, 0.10139860139860139, 0.5139860139860138]
        for i, exp in enumerate(expected):
            assert math.isclose(beta_dist_max_prob(alpha, beta, i), exp)
