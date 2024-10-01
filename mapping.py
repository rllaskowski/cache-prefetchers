
import mmh3

class Mapping:
    def __init__(self, max_size: int, hash_mapping: bool = False):
        self.max_size = max_size
        self.hash_mapping = hash_mapping
        self.mapping: dict[int, int] = {}

    def map(self, key):
        if isinstance(key, list) or isinstance(key, tuple):
            return self._map_list(key)

        if self.hash_mapping:
            return mmh3.hash(str(key)) % self.max_size if self.max_size > 0 else 0

        if key not in self.mapping:
            self.mapping[key] = min(len(self.mapping), self.max_size - 1)

        return self.mapping[key]

    def _map_list(self, keys):
        return [self.map(x) for x in keys]

    def __call__(self, key):
        return self.map(key)

    def __len__(self):
        return self.max_size
