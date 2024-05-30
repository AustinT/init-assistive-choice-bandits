from collections import Counter

import numpy as np
import pytest

from ind_acb.alg import RUCB


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
        tol = 150
        assert all(expected - tol < count < expected + tol for count in choice_counts.values())

        # Check: no predicted winner
        assert alg.predict_winner([0, 1, 2, 3]) is None

    def test_expected_duel(self):
        """
        A scenario where W =
        100    0    0
         50  100   10
        100   90  100

        At large time, the duel (a1, a2) should be chosen.
        """

        alg = RUCB()
        alg.W = np.array([[100, 0, 0], [50, 100, 10], [100, 90, 100]])

        choice = alg.choose_action_set(1e7, [0, 1, 2])  # NOTE: time is not accurate here
        assert sorted(choice) == [1, 2]

        # Check: arm 2 is predicted to win
        assert (
            alg.predict_winner(
                [
                    0,
                    1,
                    2,
                ]
            )
            == 2
        )

    @pytest.mark.parametrize(
        "utilities",
        [
            [1, 0, 0],  # larger differences
            [0.5, 0.4, 0.3],  # small difference
            [1, 0, 0, 0, 0],  # larger set, large difference
        ],
    )
    def test_best_arm_found(self, utilities: list[float]):
        alg = RUCB()
        rng = np.random.default_rng(0)
        arms = list(range(len(utilities)))
        for t in range(100_000):
            t += 1  # time starts at 1
            A_t = alg.choose_action_set(t, arms)
            chosen_idx = 0 if rng.random() < sigmoid(utilities[A_t[0]] - utilities[A_t[1]]) else 1
            alg.record_action(t, A_t, A_t[chosen_idx])
        assert alg.predict_winner(arms) == np.argmax(utilities)
