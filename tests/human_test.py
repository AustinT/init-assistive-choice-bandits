import math
from collections import Counter

from ind_acb import human


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
        # Record 2 different rewards
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
