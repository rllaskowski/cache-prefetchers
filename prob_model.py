import abc
from collections import deque
from typing import List, Set, Tuple

from cache import Cache
import torch
from collections import deque
from fieldfm import FieldAwareFactorizationMachine, train_ffm, FieldAwareFactorizationMachineModel
from typing import List, Set, Tuple, Optional, Dict
from torch import optim


class ProbablistModel(abc.ABC):
    def __init__(self, name):
        self.name = name

    def get_least_probable(self, p=1):
        pass

    def get_most_probable(self, p=1):
        pass

    def on_read(self, address, is_hit):
        pass

    def train(self, cache: Cache):
        pass


class Markov(ProbablistModel):
    def __init__(self, t):
        super().__init__("Markov")
        self.t = t
        self.transitions = {}
        self.current_state = deque(maxlen=t)

    def update_transition(self):
        if len(self.current_state) == self.t:
            state = tuple(self.current_state)
            self.transitions[state] = self.transitions.get(state, 0) + 1

    def get_least_probable(self, p=1):
        sorted_transitions = sorted(self.transitions.items(), key=lambda x: x[1])
        return [transition[0][-1] for transition in sorted_transitions[:p]]

    def get_most_probable(self, p=1):
        sorted_transitions = sorted(self.transitions.items(), key=lambda x: x[1], reverse=True)
        return [transition[0][-1] for transition in sorted_transitions[:p]]

    def on_read(self, address, is_hit):
        self.current_state.append(address)
        self.update_transition()


class FFM(ProbablistModel):
    def __init__(
        self, n, h=14, k=5, epochs=1, lr=0.01, wd=0.1, epoch_samples=20, my_ffm=False, allow_non_eviction=False, with_neutral=False
    ):
        super().__init__("FFM")

        if my_ffm:
            self.ffm = FieldAwareFactorizationMachine(n, h + 2, k)
        else:
            self.ffm = FieldAwareFactorizationMachineModel([n]*(h+2), k)
        self.h = h
        self.n = n
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.epoch_samples = epoch_samples
        self.occurences = {}
        self.losses = []
        self.optimizer = optim.AdamW(self.ffm.parameters(), lr=lr, weight_decay=wd)
        self.with_neutral = with_neutral

    def get_probability(self, cache: Cache, a: int, b: int) -> float:
        a = a % self.n
        b = b % self.n

        current_state = [x[0] % self.n for x in cache.access_history[-self.h :]]
        x = [a, b] + current_state

        return float(self.ffm(x).item())

    def get_probabilites(self, cache: Cache, ab: List[Tuple[int, int]]) -> List[float]:
        if len(cache.access_history) <= self.h:
            return [0.5] * len(ab)

        current_state = [x[0] % self.n for x in cache.access_history[-self.h :]]
        x = [[a, b] + current_state for a, b in ab]
        with torch.no_grad():
            self.ffm.eval()
            return [float(p) for p in self.ffm(x)]

    def on_read(self, cache, address, is_hit):
        i = len(cache.access_history) - 1
        address = address % self.n
        if address not in self.occurences:
            self.occurences[address] = deque()

        self.occurences[address].append(i)

    def train(self, cache: Cache):
        losses = train_ffm(
            self.ffm,
            optimizer=self.optimizer,
            history=[x[0] for x in cache.access_history],
            cache_history=cache.history,
            h=self.h,
            epochs=self.epochs,
            epoch_samples=self.epoch_samples,
            occurences=self.occurences,
            with_neutral=self.with_neutral,
        )
        self.losses.extend(losses)
