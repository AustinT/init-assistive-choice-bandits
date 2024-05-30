"""Code for recommendation algorithms."""
import math
import random
from abc import ABC, abstractmethod

import numpy as np


class BaseAlgPolicy(ABC):
    def __init__(self, Q: int = 2):
        super().__init__()
        self.Q = Q
        self.reset()
        self.rng = random.Random()

    @abstractmethod
    def choose_action_set(self, time: int, all_actions: list[int]) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def record_action(self, time: int, available_actions: list[int], chosen_action: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class RUCB(BaseAlgPolicy):
    def reset(self):
        assert self.Q == 2, "Only for Q=2"
        self.W = None
        self.alpha = 1.0
        self.B = set()

    def choose_action_set(self, time: int, all_actions: list[int]):
        # Init W
        if self.W is None:
            self.W = np.zeros((len(all_actions), len(all_actions)), dtype=int)

        # Create U matrix
        W_plus_W_T = self.W + self.W.T
        with np.errstate(invalid="ignore"):
            U = self.W / W_plus_W_T + np.sqrt(self.alpha * math.log(time) / W_plus_W_T)
        assert not np.any(np.isinf(U))  # just NaNs
        U = np.nan_to_num(U, nan=1.0)
        for i in range(len(U)):
            U[i, i] = 0.5  # fill exact diagonal

        # Create C set
        C = set()
        for i, _ in enumerate(all_actions):
            if np.all(U[i] >= 0.5):
                C.add(i)  # append action idxs
        if len(C) == 0:
            C = {int(self.rng.choice(all_actions))}

        # B set
        self.B = self.B & C

        # Pick a_c
        if len(C) == 1:
            a_c = list(C)[0]
            self.B = set(C)
        else:
            if self.rng.random() < 0.5 and len(self.B) > 0:
                a_c = list(self.B)[0]
            else:
                # sample an item from C, not in B
                a_c = int(self.rng.choice([x for x in C if x not in self.B]))

        # Pick arm d with correct tie breaking
        max_u = np.max(U[:, a_c])
        d_candidates = set(np.where(U[:, a_c] >= max_u)[0])
        if len(d_candidates) > 1:
            d_candidates.discard(a_c)  # d =/= c for ties

        # Return action
        return [a_c, int(self.rng.choice(list(d_candidates)))]

    def record_action(self, time: int, available_actions: list[int], chosen_action: int) -> None:
        assert len(available_actions) == 2

        # Find out which action was not chosen.
        # Guess the first one, and move to the second if equal.
        # Handles case of actions being equal
        not_chosen_action = available_actions[0]
        if not_chosen_action == chosen_action:
            not_chosen_action = available_actions[1]
        self.W[chosen_action, not_chosen_action] += 1
