import abc
from collections import OrderedDict, deque
from typing import List, Set
from cache import Cache
import scipy.optimize as opt
import random


class EvictionStrategy(abc.ABC):
    """Manages cache page eviction."""

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def _evict(self, cache: Cache, p=1) -> Set[int]:
        """Returns a set of addresses to evict from the cache."""
        pass


    def evict(self, cache: Cache, p=1):
        """Evicts p pages from the cache."""
        pages = set(self._evict(cache, p))

        assert len(pages) == p,\
            f'{self.name} eviction strategy returned wrong number of unique addresses'

        assert pages.issubset(cache.cache),\
            f'{self.name} eviction strategy returned addresses not in cache'

        cache.cache -= pages


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
    def __init__(self, max_frequency=5):
        super().__init__("MQ")
        self.max_frequency = max_frequency
        self.frequency_queues = {i: deque() for i in range(max_frequency)}
        self.page_frequency = {}

    def read_callback(self, address, is_hit):
        if is_hit:
            # Update the frequency of the page if it's a hit
            current_frequency = self.page_frequency.get(address, 0)
            self.frequency_queues[current_frequency].remove(address)
            new_frequency = min(current_frequency + 1, self.max_frequency - 1)
            self.frequency_queues[new_frequency].append(address)
            self.page_frequency[address] = new_frequency
        else:
            # If it's a miss and the address is new, add it to the lowest frequency queue
            if address not in self.page_frequency:
                self.frequency_queues[0].append(address)
                self.page_frequency[address] = 0

    def fetch_callback(self, addresses):
        for address in addresses:
            # If the address is new, add it to the lowest frequency queue
            if address not in self.page_frequency:
                self.frequency_queues[0].append(address)
                self.page_frequency[address] = 0

    def get_to_evict(self, cache, p=1):
        to_evict = set()
        for _ in range(p):
            # Evict from the lowest non-empty frequency queue
            for _, queue in self.frequency_queues.items():
                if queue:
                    evict_address = queue.popleft()
                    # Remove the address from page_frequency mapping
                    del self.page_frequency[evict_address]
                    to_evict.add(evict_address)
                    break
            if not to_evict:
                # No more pages to evict
                break
        return to_evict



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
        # Assume prob_model can provide expected time until next request for each page
        page_times = {page: self.prob_model.get_expected_time(page) for page in cache.cache}
        # Sort pages by expected time (longest first)
        sorted_pages = sorted(page_times, key=page_times.get, reverse=True)
        return sorted_pages[:p]

    def read_callback(self, address, is_hit):
        # Update the probabilistic model on every read
        self.prob_model.on_read(address, is_hit)


def solve_linear_program(probabilities, pages):
    # Number of pages
    n = len(pages)

    # Objective function: maximize the sum of probabilities (all coefficients are 1)
    c = [-1] * n

    # Constraints
    A_ub = []  # Coefficient matrix for inequalities
    b_ub = []  # Constraint bounds for inequalities

    # Building constraints based on p(a, b)
    for i, page_a in enumerate(pages):
        for j, page_b in enumerate(pages):
            if page_a != page_b:
                constraint = [0] * n
                constraint[i] = probabilities[(page_a, page_b)]
                constraint[j] = 1 - probabilities[(page_a, page_b)]
                A_ub.append(constraint)
                b_ub.append(0.5)

    # Bounds for each variable (probability) between 0 and 1
    bounds = [(0, 1)] * n

    # Sum of probabilities must be 1
    A_eq = [[1] * n]
    b_eq = [1]

    # Solve the linear programming problem
    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    return result.x


class DOM(EvictionStrategy):

    def __init__(self, prob_model):
        super().__init__("DOM")
        self.prob_model = prob_model

    def _evict(self, cache: Cache, p=1) -> Set[int]:
        if len(cache.cache) <= 1:
            return []
    
        probabilities = {}
        input = []
        for a in cache.cache:
            for b in cache.cache:
                if a != b:
                    input.append((a,b))

        for ab, prob in zip(input, self.prob_model.get_probabilites(cache, input)):
            probabilities[ab] = prob

        distribution = solve_linear_program(probabilities, list(cache.cache))

        sampled_pages = random.choices(list(cache.cache), weights=distribution, k=p)

        self.prob_model.train(cache)

        return set(sampled_pages)

    def read_callback(self, cache, address, is_hit):
        self.prob_model.on_read(address, is_hit)
