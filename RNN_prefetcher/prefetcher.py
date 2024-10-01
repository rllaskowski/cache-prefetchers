from collections import deque


class StreamPrefetcher:
    def __init__(self, prefetch_distance, max_prefetched_addresses):
        self.prefetch_distance = prefetch_distance
        self.prefetched_addresses = set()
        self.prefetched_queue = deque(maxlen=max_prefetched_addresses)

    def prefetch(self, address):
        next_address = address + self.prefetch_distance
        if len(self.prefetched_addresses) == self.prefetched_queue.maxlen:
            oldest_address = self.prefetched_queue.popleft()
            self.prefetched_addresses.remove(oldest_address)

        self.prefetched_addresses.add(next_address)
        self.prefetched_queue.append(next_address)

    def access(self, address):
        if address in self.prefetched_addresses:
            self.prefetched_addresses.remove(address)
            self.prefetched_queue.remove(address)
            return True
        else:
            self.prefetch(address)
            return False
