from collections import Counter

import numpy as np

from ind_acb.alg import RUCB


class TestRUCB:
    """Try to test RUCB."""

    def test_first_action_uniform(self):
        """First action should be a uniformly random duel."""

        # Choose the first action many times
        choices = []
        for _ in range(10000):
            alg = RUCB()
            choices.append(alg.choose_action_set(1, [0, 1, 2, 3]))

        # Merge identical action sets
        choices = [tuple(sorted(c)) for c in choices]
        choice_counts = Counter(choices)

        # Check that all action sets are chosen roughly equally
        assert len(choice_counts) == 6  #  (4 choose 2)
        expected = 1667  # 10000 / 6 = 1666.66...
        tol = 100
        assert all(expected - tol < count < expected + tol for count in choice_counts.values())

    def test_expected_duel(self):
        """
        A scenario where W =
        100    0    0
          5  100   30
        100   70  100

        Here, only a2 should have a decent chance of being the Condorcet winner,
        so it should be chosen as ac.

        Among opponents, a1 should be chosen, since it has a 10% win rate against a2.
        """

        alg = RUCB()
        alg.W = np.array([[100, 0, 0], [5, 100, 30], [100, 70, 1000]])

        choice = alg.choose_action_set(100, [0, 1, 2])  # NOTE: time is not accurate here
        assert choice == [2, 1]
