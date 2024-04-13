import abc
import heapq
from collections import deque
from typing import Tuple


class PrefetchStrategy(abc.ABC):
    """Manages cache page prefetching."""

    def __init__(self, name):
        self.name = name

    def get_to_fetch(self, cache, p=1):
        pass

    def read_callback(self, cache, address, is_hit):
        pass


class Next(PrefetchStrategy):
    def __init__(self):
        super().__init__("NEXT")

    def get_to_fetch(self, cache, p=1):
        last_addr, _ = cache.access_history[-1]
        return [last_addr + i for i in range(1, p + 1)]


class Markov(PrefetchStrategy):
    def __init__(self, t=3):
        super().__init__("MARKOV")
        self.t = t
        self.transitions = {}
        self.current_state = deque(maxlen=t + 1)

    def get_to_fetch(self, cache, p=1):
        assert p == 1, "For now this works only for p=1"

        if tuple(self.current_state) not in self.transitions:
            # This will happend at the begging of cache work
            # just return the next address like in NEXT
            last_addr, _ = cache.access_history[-1]
            return [last_addr + 1]

        prev = tuple(self.current_state)

        transitions = self.transitions[prev]

        next = max(transitions, key=transitions.get)

        return [next[-1]]

    def read_callback(self, _cache, address, is_hit):
        prev = tuple(self.current_state)

        self.current_state.append(address)
        current = tuple(self.current_state)

        self.update_transition(prev, current)

    def update_transition(self, prev: Tuple, current: Tuple):
        if len(prev) != self.t + 1:
            return

        if prev not in self.transitions:
            self.transitions[prev] = {}

        if current not in self.transitions[prev]:
            self.transitions[prev][current] = 0

        self.transitions[prev][current] += 1


class EnsamblePrefetcher(PrefetchStrategy):
    def __init__(self, prefetchers):
        super().__init__("Ensamble")
        self.prefetchers = prefetchers

    def get_to_fetch(self, cache, p):
        assert p == len(
            self.prefetchers
        ), "Ensamble prefetcher works only for p == len(prefetchers)"

        to_fetch = []
        for prefetcher in self.prefetchers:
            to_fetch.extend(prefetcher.get_to_fetch(cache, 1))

        return set(to_fetch)

    def read_callback(self, cache, address, is_hit):
        for prefetcher in self.prefetchers:
            prefetcher.read_callback(cache, address, is_hit)
