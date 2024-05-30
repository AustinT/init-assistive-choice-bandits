import math
from collections import Counter

from ind_acb import human
from ind_acb.misc import beta_dist_max_prob


def _exact_prob_matches(policy, time: int, a1: int, a2: int, N: int = 10_000):
    expected_prob = policy.action_choice_prob(time, a1, a2)
    observed_choices = Counter([policy.choose_action(time, [a1, a2]) for _ in range(N)])
    observed_prob = observed_choices[a1] / N
    assert abs(expected_prob - observed_prob) < 0.05


class BaseHumanTest:
    def _make_policy(self) -> human.BaseHumanPolicy:
        raise NotImplementedError

    def test_smoke(self):
        """Make the policy, get some rewards."""
        pol = self._make_policy()
        a = pol.choose_action(1, [0, 1])
        pol.record_reward(1, a, 1.0)
        pol.reset()


class TestHumanUCB(BaseHumanTest):
    def _make_policy(self) -> human.HumanUCB:
        policy = human.HumanUCB()
        # Record 3 different rewards
        policy.record_reward(1, 0, 1.0)
        policy.record_reward(2, 1, 0.0)
        policy.record_reward(3, 0, 0.0)
        return policy

    def test_ucb_values(self):
        policy = self._make_policy()

        # Test several hand-calculated UCB value at time 4
        assert math.isclose(policy._get_ucb(4, action=0), 0.5 + math.sqrt(math.log(4) / 2))
        assert math.isclose(policy._get_ucb(4, action=1), math.sqrt(math.log(4)))
        assert math.isclose(policy._get_ucb(4, action=2), math.inf)

    def test_exact_prob(self):
        policy = self._make_policy()
        _exact_prob_matches(policy, time=4, a1=0, a2=1)
        _exact_prob_matches(policy, time=4, a1=0, a2=2)
        _exact_prob_matches(policy, time=4, a1=1, a2=2)

    def test_choose_action(self):
        policy = self._make_policy()

        # Test that the policy chooses the action with the highest UCB value
        assert policy.choose_action(4, [0, 1]) == 0
        assert policy.choose_action(4, [0, 1, 2]) == 2
        assert policy.choose_action(4, [1, 2]) == 2


class TestHumanThompsonSampling(BaseHumanTest):
    def _make_policy(self) -> human.HumanThompsonSampling:
        policy = human.HumanThompsonSampling([1, 2, 3], [4, 5, 6])

        # Record 3 different rewards
        policy.record_reward(1, 0, 1.0)
        policy.record_reward(2, 1, 0.0)
        policy.record_reward(3, 0, 0.0)

        # After this, posterior should be
        # alpha = [2, 2, 3]
        # beta = [5, 6, 6]
        return policy

    def test_exact_prob(self):
        policy = self._make_policy()
        _exact_prob_matches(policy, time=4, a1=0, a2=1)
        _exact_prob_matches(policy, time=4, a1=0, a2=2)
        _exact_prob_matches(policy, time=4, a1=1, a2=2)

    def test_choose_action(self):
        policy = self._make_policy()

        # Choose many random actions
        actions = [policy.choose_action(4, [0, 1, 2]) for _ in range(10000)]

        # Check that the distribution of actions matches the expected distribution
        action_counts = Counter(actions)
        for i, count in action_counts.items():
            observed_freq = count / len(actions)
            expected_freq = beta_dist_max_prob([2, 2, 3], [5, 6, 6], i)
            assert math.isclose(observed_freq, expected_freq, abs_tol=0.05)
