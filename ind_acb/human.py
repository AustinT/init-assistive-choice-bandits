"""Code for human policies."""
import math
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

from ind_acb.misc import beta_dist_max_prob


class BaseHumanPolicy(ABC):
    def __init__(self):
        super().__init__()
        self.reset()

    @abstractmethod
    def choose_action(self, time: int, available_actions: list[int]) -> int:
        pass

    @abstractmethod
    def record_reward(self, time: int, action: int, reward: float) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class ExactProbHumanMixin(ABC):
    """Mixin for human policies which can compute their probability of taking each action."""

    @abstractmethod
    def action_choice_prob(self, time: int, a1: int, a2: int) -> float:
        """Special case for 2 actions: probability of choosing action 1 over action 2."""
        pass


class HumanUCB(BaseHumanPolicy, ExactProbHumanMixin):
    def __init__(self, ucb_constant: float = 1.0):
        super().__init__()
        self.ucb_constant = ucb_constant

    def reset(self):
        self.action_to_count = dict()
        self.action_to_mean = dict()

    def _get_ucb(self, time: int, action: int) -> float:
        return self.action_to_mean.get(action, math.inf) + self.ucb_constant * math.sqrt(
            math.log(time) / self.action_to_count.get(action, 1)
        )

    def choose_action(self, time: int, available_actions: list[int]) -> int:
        argmax = np.argmax([self._get_ucb(time, a) for a in available_actions])
        return available_actions[int(argmax)]

    def record_reward(self, time: int, action: int, reward: float) -> None:
        old_count = self.action_to_count.get(action, 0)
        self.action_to_mean[action] = (self.action_to_mean.get(action, 0) * old_count + reward) / (old_count + 1)
        self.action_to_count[action] = old_count + 1

    def action_choice_prob(self, time: int, a1: int, a2: int) -> float:
        ucb1 = self._get_ucb(time, a1)
        ucb2 = self._get_ucb(time, a2)
        if ucb1 == ucb2:
            return 0.5
        elif ucb1 > ucb2:
            return 1.0
        else:
            return 0.0


class HumanThompsonSampling(BaseHumanPolicy, ExactProbHumanMixin):
    def __init__(self, alpha0: np.ndarray, beta0: np.ndarray):
        self.alpha0 = np.asarray(alpha0)
        self.beta0 = np.asarray(beta0)
        super().__init__()

    def reset(self):
        self.alpha = self.alpha0.copy()
        self.beta = self.beta0.copy()

    def choose_action(self, time: int, available_actions: list[int]) -> int:
        sample = stats.beta(self.alpha, self.beta).rvs()
        return max(available_actions, key=lambda a: sample[a])

    def record_reward(self, time: int, action: int, reward: float) -> None:
        if reward == 1.0:
            self.alpha[action] += 1
        elif reward == 0.0:
            self.beta[action] += 1
        else:
            raise ValueError(reward)

    def action_choice_prob(self, time: int, a1: int, a2: int) -> float:
        # Note: "0" below is because a1 is in index 0
        return beta_dist_max_prob(self.alpha[[a1, a2]], self.beta[[a1, a2]], 0)
