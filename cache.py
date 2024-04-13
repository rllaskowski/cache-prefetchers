from calendar import c
from typing import List, Set, Tuple
import copy

class Cache:
    def __init__(self, name: str, eviction_strategy, prefetch_strategy=None, p=1, size=8):
        self.name = name
        self.eviction_strategy = eviction_strategy
        self.prefetch_strategy = prefetch_strategy
        self.p = p  # prefetch degree
        self.access_history: List[Tuple[int, int]] = []  # [(address, is_hit)]
        self.cache: Set[int] = set()
        self.size = size
        self.history = []

    def evict(self, p=1, allow_non_eviction=False):
        n_evicted = self.eviction_strategy.evict(self, p)
        if n_evicted is None and not allow_non_eviction:
            raise ValueError(f"{self.eviction_strategy.name} eviction strategy returned None")

        return n_evicted


    def prefetch(self, p=1):
        to_fetch = set(self.prefetch_strategy.get_to_fetch(self, p))

        new_fetched = to_fetch - self.cache.intersection(to_fetch)

        if len(new_fetched) == 0:
            return

        for _ in range(len(self.cache) + len(new_fetched) - self.size):
            self.evict(1)

        self.eviction_strategy.fetch_callback(self, new_fetched)

        self.cache |= set(new_fetched)

    def read(self, address):
        hit = address in self.cache

        self.access_history.append((address, hit))
        self.history.append(copy.deepcopy(self.cache))

        if not hit:
            if len(self.cache) == self.size:
                n_evicted = self.evict(1, allow_non_eviction=True)
            

            self.cache.add(address)

        self.eviction_strategy.read_callback(self, address, hit)

        if self.prefetch_strategy:
            self.prefetch_strategy.read_callback(self, address, hit)

        if self.prefetch_strategy is not None:
            self.prefetch(self.p)

        assert len(self.cache) <= self.size, f"{self.name} cache size exceeded"

        return hit

    def __str__(self):
        return f"{self.name} cache: {self.cache}"
