from typing import List, Set, Tuple


class Cache:

    def __init__(
        self,
        name: str,
        eviction_strategy,
        prefetch_strategy=None,
        p=1,
        size=8
    ):
        self.name = name
        self.eviction_strategy = eviction_strategy
        self.prefetch_strategy = prefetch_strategy
        self.p = p  # prefetch degree
        self.access_history: List[Tuple[int, int]] = [] # [(address, is_hit)]
        self.cache: Set[int] = set()
        self.size = size

    def evict(self, p=1):
        self.eviction_strategy.evict(self, p)


    def prefetch(self, p=1):
        if len(self.cache) == self.size:
            self.evict(self.p)

        to_fetch = self.prefetch_strategy.get_to_fetch(self, p)

        assert len(set(to_fetch)) == p,\
            f'{self.prefetch_strategy.name} prefetch strategy returned wrong number of addresses'

        self.eviction_strategy.fetch_callback(self, to_fetch)

        self.cache |= set(to_fetch)


    def read(self, address):
        hit = address in self.cache

        self.access_history.append((address, hit))

        if not hit:
            if len(self.cache) == self.size:
                self.evict(1)

            self.cache.add(address)

        self.eviction_strategy.read_callback(self, address, hit)

        if self.prefetch_strategy:
            self.prefetch_strategy.read_callback(self, address, hit)

        if self.prefetch_strategy is not None:
            self.prefetch(self.p)

        assert len(self.cache) <= self.size, f'{self.name} cache size exceeded'

        return hit

    def __str__(self):
        return f'{self.name} cache: {self.cache}'
