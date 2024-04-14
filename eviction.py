import abc
import random
from collections import OrderedDict, defaultdict, deque
from typing import List, Set, Optional, Tuple
import math
import scipy.optimize as opt
import numpy as np
import math



from cache import Cache

ASSERTS_ENABLED = True


class EvictionStrategy(abc.ABC):
    """Manages cache page eviction."""

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def _evict(self, cache: Cache, p=1) -> Set[int]:
        """Returns a set of addresses to evict from the cache."""
        pass

    def evict(self, cache: Cache, p=1) -> Optional[int]:
        """Evicts p pages from the cache."""
        pages = set(self._evict(cache, p))

        if pages == None:
            return None

        assert (
            len(pages) == p
        ), f"{self.name} eviction strategy returned wrong number of unique addresses"

        assert pages.issubset(
            cache.cache
        ), f"{self.name} eviction strategy returned addresses not in cache"

        cache.cache -= pages

        return pages

    def fetch_callback(self, cache: Cache, addresses: List[int]):
        """Called by cache when a page is fetched to the cache."""
        pass

    def read_callback(self, cache: Cache, address: int, is_hit: bool):
        """Called cache when a page is read from the cache."""
        pass


class LRU(EvictionStrategy):
    def __init__(self):
        super().__init__("LRU")
        self.accesses = OrderedDict()

    def fetch_callback(self, _cache, addresses):
        for address in addresses:
            self.accesses[address] = None
            self.accesses.move_to_end(address)

    def read_callback(self, _cache, address, is_hit):
        if is_hit:
            self.accesses.move_to_end(address)
        else:
            self.accesses[address] = None
            self.accesses.move_to_end(address)

    def _evict(self, _cache: Cache, p=1):
        to_evict = set()

        for _ in range(p):
            to_evict.add(self.accesses.popitem(last=False)[0])

        return to_evict



class MQ(EvictionStrategy):
    def __init__(self, max_frequency=200, qout_size=100, life_time=1000):
        super().__init__("MQ")
        self.max_frequency = max_frequency
        self.frequency_queues = {i: deque() for i in range(max_frequency)}
        self.qout = deque(maxlen=qout_size)
        self.page_frequency = {}
        self.page_queue_num = {}
        self.page_expire_time = {}
        self.life_time = life_time
        self.current_time = 0

    def queue_num(self, frequency):
        return min(int(math.log(frequency + 1, 2)) - 1, self.max_frequency - 1)

    def read_callback(self, _cache, address, is_hit):
        self.current_time += 1
        self.adjust()

        if is_hit:
            queue_num = self.page_queue_num[address]
            self.frequency_queues[queue_num].remove(address)
            current_frequency = self.page_frequency[address] + 1

            assert address not in self.qout, "address is in qout, but it shouldnt"
        else:
            if address in self.qout:
                self.qout.remove(address)
                current_frequency = self.page_frequency[address] + 1
            else:
                current_frequency = 1

        queue_num = self.queue_num(current_frequency)

        self.frequency_queues[queue_num].append(address)
        self.page_frequency[address] = current_frequency
        self.page_queue_num[address] = queue_num
        self.page_expire_time[address] = self.current_time + self.life_time

        if ASSERTS_ENABLED:
            for address in _cache.cache:
                assert address in self.page_frequency, "address is not in page_frequency"
                assert address in self.page_queue_num, "address is not in page_queue_num"
                assert address in self.page_expire_time, "address is not in page_expire_time"
                assert address not in self.qout, "address is in qout"

    def fetch_callback(self, _cache, addresses):
        for address in addresses:
            if address in self.page_queue_num:
                continue

            if address in self.qout:
                self.qout.remove(address)
                current_frequency = self.page_frequency[address] + 1
            else:
                current_frequency = 1

            queue_num = self.queue_num(current_frequency)
            self.frequency_queues[queue_num].append(address)
            self.page_queue_num[address] = queue_num
            self.page_frequency[address] = current_frequency

            self.page_expire_time[address] = self.current_time + self.life_time

        if ASSERTS_ENABLED:
            for address in _cache.cache:
                assert address in self.page_frequency, "address is not in page_frequency"
                assert address in self.page_queue_num, "address is not in page_queue_num"
                assert address in self.page_expire_time, "address is not in page_expire_time"
                assert address not in self.qout, "address is in qout"

    def _evict(self, _cache: Cache, p=1):
        to_evict = set()

        for _ in range(p):
            for _, queue in self.frequency_queues.items():
                if queue:
                    evict_address = queue.popleft()

                    if len(self.qout) >= self.qout.maxlen:
                        quout_evict = self.qout.popleft()
                        del self.page_frequency[quout_evict]

                    if ASSERTS_ENABLED:
                        assert evict_address not in self.qout, "address is already in qout"

                    self.qout.append(evict_address)

                    del self.page_expire_time[evict_address]
                    del self.page_queue_num[evict_address]

                    to_evict.add(evict_address)
                    break
            if not to_evict:
                break

        return to_evict

    def adjust(self):
        for i, queue in self.frequency_queues.items():
            if i == 0 or not queue:
                continue

            front_address = queue[0]

            if self.page_expire_time[front_address] < self.current_time:
                queue.popleft()
                self.page_expire_time[front_address] = self.current_time + self.life_time
                new_queue_num = i - 1
                self.frequency_queues[new_queue_num].append(front_address)
                self.page_queue_num[front_address] = new_queue_num


class MP(EvictionStrategy):
    def __init__(self, prob_model):
        super().__init__("MP")
        self.prob_model = prob_model

    def evict(self, cache, p=1):

        most_probable = self.prob_model.get_most_probable(len(cache.cache))

        to_evict = [page for page in cache.cache if page not in most_probable]
        return to_evict[:p]

    def read_callback(self, address, is_hit):

        self.prob_model.on_read(address, is_hit)


class MET(EvictionStrategy):
    def __init__(self, prob_model):
        super().__init__("MET")
        self.prob_model = prob_model

    def get_to_evict(self, cache, p=1):
        page_times = {page: self.prob_model.get_expected_time(page) for page in cache.cache}
        sorted_pages = sorted(page_times, key=page_times.get, reverse=True)
        return sorted_pages[:p]

    def read_callback(self, address, is_hit):
        self.prob_model.on_read(address, is_hit)


def solve_linear_program(probabilities, pages):
    for a, b in probabilities.keys():
        assert math.isclose(probabilities[(a, b)] + probabilities[(b, a)], 1)

    n = len(pages)

    # Objective function
    c = [0] * n

    # Constraints
    A_ub = []  # Coefficient matrix for inequalities
    b_ub = []  # Constraint bounds for inequalities

    # Building constraints based on p(a, b)
    for v, page_v in enumerate(pages):
        constraint = [0] * n
        for u, page_u in enumerate(pages):
            if page_v == page_u:
                continue

            constraint[u] = probabilities[(page_v, page_u)]
            c[u] += probabilities[(page_v, page_u)]

        A_ub.append(constraint)
        b_ub.append(0.5)

    # Bounds for each variable (probability) between 0 and 1
    bounds = [(0, 1)] * n

    # Sum of probabilities must be 1
    A_eq = [[1] * n]
    b_eq = [1]

    # Solve the linear programming problem
    result = opt.linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    return result.x


class DOM(EvictionStrategy):
    def __init__(self, prob_model, train_interval=50):
        super().__init__("DOM")
        self.prob_model = prob_model
        self.train_interval = train_interval
        self.evicted = defaultdict(int)

    def _evict(self, cache: Cache, p=1) -> Set[int]:
        if len(cache.cache) <= 1:
            return []

        probabilities = {}
        inputs = []
        for a in cache.cache:
            for b in cache.cache:
                if a != b:
                    inputs.append((a, b))

        for ab, prob in zip(inputs, self.prob_model.get_probabilites(cache, inputs)):
            probabilities[ab] = prob

        for a, b in inputs:
            if a < b:
                probabilities[(a, b)] = 1 - probabilities[(b, a)]
            elif a == b:
                del probabilities[(a, b)]

        cache_list = list(cache.cache)
        distribution = solve_linear_program(probabilities, cache_list)

        sampled_pages = random.choices(cache_list, weights=distribution, k=p)

        if len(cache.access_history) % self.train_interval == 0:
            self.prob_model.train(cache)

        evicted = sampled_pages[0]
        self.evicted[evicted] += 1

        return set(sampled_pages)

    def read_callback(self, cache, address, is_hit):
        self.prob_model.on_read(cache, address, is_hit)


class RandomEvictor(EvictionStrategy):
    def __init__(self):
        super().__init__("RANDOM")

    def _evict(self, cache: Cache, p=1) -> Set[int]:
        return set(random.sample(cache.cache, p))
