import abc
from collections import deque
from typing import List, Set, Tuple

from cache import Cache
from ffm import FieldAwareFactorizationMachine, train_ffm


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

    def __init__(self, n, h=14, k=5, epochs=1, lr=0.01, wd=0.1, epoch_samples=4):
        super().__init__("FFM")

        self.ffm = FieldAwareFactorizationMachine(n, h+2, k)
        self.h = h
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.epoch_samples = epoch_samples

        self.losses = []

    def get_probability(self, cache: Cache, a: int, b: int) -> float:
        a = a%self.ffm.n
        b = b%self.ffm.n

        current_state = [x[0] % self.ffm.n for x in cache.access_history[-self.h:]]
        x = [a, b]+current_state

        return float(self.ffm(x).item())

    def get_probabilites(self, cache: Cache, ab: List[Tuple[int, int]]) -> List[float]:
        if len(cache.access_history) <= self.h:
            return [0.5] * len(ab)

        current_state = [x[0] % self.ffm.n for x in cache.access_history[-self.h:]]
        x = [
            [a%self.ffm.n, b%self.ffm.n] + current_state
            for a, b in ab
        ]

        return [float(p) for p in self.ffm(x)]

    def train(self, cache: Cache):
        losses = train_ffm(
            self.ffm,
            history=[x[0] % self.ffm.n for x in cache.access_history],
            cache=set([p % self.ffm.n for p in cache.cache]),
            h=self.h,
            epochs=self.epochs,
            lr=self.lr,
            wd=self.wd,
            epoch_samples=self.epoch_samples
        )
        self.losses.extend(losses)
