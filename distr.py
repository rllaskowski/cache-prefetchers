from calendar import c
from copy import deepcopy
from math import e
from collections import defaultdict
import re
from IPython import embed
from sympy import li
import torch
import torch.nn as nn
from traitlets import default
from zmq import device
import numpy as np
import random
import bisect
import copy
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequenceModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.delta_embedding = nn.Embedding(input_size, hidden_size)

        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=2)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)

        x = x.to(device)

        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        output = self.fc(output)
        output = self.softmax(output)
        return output


class ConstantSequenceModel(nn.Module):
    def __init__(
        self,
        n_pages,
        cache_size,
        history_size,
        emb_size,
        delta_emb_size,
        hidden_size,
        output_size,
        n_deltas=100,
        dropout=0.1,
    ):
        super().__init__()
        self.query_emb = self.key_emb = emb_size + delta_emb_size
        self.delta_emb_size = delta_emb_size
        self.emb_size = emb_size
        self.cache_size = cache_size
        self.history_size = history_size
        self.embedding = nn.Embedding(n_pages, emb_size)
        self.delta_embedding = nn.Embedding(n_deltas, delta_emb_size)
        self.fc1 = nn.Linear(
            self.cache_size
            + (self.emb_size + self.delta_emb_size)
            + self.cache_size * (self.emb_size + self.delta_emb_size)
            + (self.emb_size + self.delta_emb_size),
            hidden_size,
        )
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.query = nn.Linear(emb_size + delta_emb_size, self.query_emb)
        self.key = nn.Linear(emb_size + delta_emb_size, self.key_emb)
        self.softmax = nn.Softmax(dim=1)
        self.output_size = output_size
        self.n_elements = n_pages
        self.rnn = nn.GRU(emb_size + delta_emb_size, emb_size + delta_emb_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.to(device)

        for layer in [self.fc1, self.fc2, self.query, self.key]:
            torch.nn.init.xavier_uniform_(layer.weight.data)

    def _to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)
        return x.to(device)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.query_emb, dtype=torch.float)
        )
        scores = self.softmax(scores)
        output = torch.matmul(scores, value)
        return output

    def forward(self, cache, cache_deltas, history, history_deltas):
        cache = self._to_tensor(cache)
        history = self._to_tensor(history)
        history_deltas = self._to_tensor(history_deltas)
        cache_deltas = self._to_tensor(cache_deltas)

        if len(cache.shape) == 1:
            cache = cache.unsqueeze(0)

        if len(history.shape) == 1:
            history = history.unsqueeze(0)

        if len(history_deltas.shape) == 1:
            history_deltas = history_deltas.unsqueeze(0)

        if len(cache_deltas.shape) == 1:
            cache_deltas = cache_deltas.unsqueeze(0)

        embedded_cache = self.embedding(cache)
        embedded_cache_deltas = self.delta_embedding(cache_deltas)
        embedded_history = self.embedding(history)
        embedded_history_deltas = self.delta_embedding(history_deltas)

        embedded_cache = torch.cat([embedded_cache, embedded_cache_deltas], dim=2)
        embedded_history = torch.cat([embedded_history, embedded_history_deltas], dim=2)

        if self.history_size > 1:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded_history, [self.history_size] * len(history), batch_first=True
            )
            _, embedded_history = self.rnn(packed)
            embedded_history = embedded_history.squeeze(0).unsqueeze(1)

        query = self.query(embedded_cache)
        key = self.key(embedded_history)
        inter = torch.matmul(query, key.transpose(-2, -1)).squeeze(-1)
        inter /= torch.sqrt(torch.tensor(self.query_emb, dtype=torch.float))
        inter_softmax = self.softmax(inter)
        weighted_sum = ((embedded_history) * inter_softmax.unsqueeze(2)).sum(dim=1)
        hidden = torch.cat(
            [
                inter.view(-1, self.cache_size),
                weighted_sum.view(-1, (self.emb_size + self.delta_emb_size)),
                embedded_cache.view(
                    -1, self.cache_size * (self.emb_size + self.delta_emb_size)
                ),
                embedded_history.view(-1, self.emb_size + self.delta_emb_size),
            ],
            dim=1,
        )
        hidden = torch.relu(self.fc1(hidden))
        output = self.fc2(hidden)
        output = self.softmax(output)

        return output.squeeze(1)


def get_training_samples(
    cache_history,
    access_history,
    cache_size,
    history_size,
    n_pages,
    occurences,
    n_deltas,
    sequence,
    cache_replacement,
    n_samples=800,
):
    samples = []

    while len(samples) < n_samples:
        T = len(cache_history)
        t = random.randint(0, T - 2)

        if len(cache_history[t]) < cache_size:
            continue

        cache = cache_history[t]

        if cache_replacement:
            to_replace = random.sample(cache, cache_replacement)
            cache = cache - set(to_replace)
            replacement = random.sample(set(range(n_pages)) - cache, cache_replacement)
            cache = cache.union(replacement)

        list_cache = sorted(list(cache))
        # cache = set(random.sample(range(n_pages), cache_size))
        history = access_history[t - history_size + 1 : t + 1]

        deltas_history = [(a - history[-1]) % n_deltas for a in history]
        cache_deltas = [(a - history[-1]) % n_deltas for a in list_cache]

        first_occ = {
            a: bisect.bisect_left(occurences[a], t + 1) for a in cache for a in cache
        }

        found = set(a for a in first_occ if first_occ[a] is not None)

        if len(found) < cache_size:
            to_evict = list(cache - found)
            assert all(a in cache for a in to_evict), "Evicted item not in cache"
        else:
            to_evict = max(found, key=lambda x: first_occ[x])
            assert to_evict in cache, "Evicted item not in cache"

        distr = [0] * len(cache)
        if isinstance(to_evict, tuple) or isinstance(to_evict, list):
            for a in to_evict:
                distr[list_cache.index(a)] = 1 / len(to_evict)
        else:
            distr[list_cache.index(to_evict)] = 1

        samples.append((list_cache, cache_deltas, history, deltas_history, distr))

    return samples


def train_model(samples, model, optimizer):
    model.train()
    cache, cache_deltas, history, history_deltas, distr = zip(*samples)

    cache = torch.tensor(cache, dtype=torch.long)
    history = torch.tensor(history, dtype=torch.long)
    distr = torch.tensor(distr, dtype=torch.float)

    optimizer.zero_grad()

    output = model(cache, cache_deltas, history, history_deltas)
    loss = nn.CrossEntropyLoss()(output, distr)
    loss.backward()
    optimizer.step()

    return loss.item()


def test_on_sequence(
    sequence,
    cache_size,
    n_elements,
    train_interval=50,
    history_size=1,
    greedy=True,
    emb_size=10,
    hidden_size=64,
    lr=1e-4,
    wd=0.1,
    train=True,
    train_samples=800,
    hash_elements=False,
    n_deltas=100,
    delta_emb_size=10,
    cache_replacement=5,
    dropout=0.1,
):
    cache = set()
    losses = []
    model = ConstantSequenceModel(
        n_pages=n_elements,
        hidden_size=hidden_size,
        emb_size=emb_size,
        output_size=cache_size,
        cache_size=cache_size,
        history_size=history_size,
        n_deltas=n_deltas,
        delta_emb_size=delta_emb_size,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    history = []
    cache_history = []
    misses = 0
    in_cache = 0
    mapping = {}
    occ = defaultdict(list)

    delta_occurences = defaultdict(int)
    cache_delta_occurences = defaultdict(int)
    misses_history = []
    delta_history = []
    pbar = tqdm.tqdm(list(enumerate(sequence)))
    for i, addr in pbar:
        if i == 0:
            delta_history.append(0)
        else:
            delta_history.append(addr - sequence[i - 1])

        if hash_elements:
            addr = addr % n_elements
        else:
            if addr not in mapping:
                mapping[addr] = len(mapping)
            addr = mapping[addr]

        occ[addr].append(i)

        if i > 0:
            delta_occurences[addr - history[-1]] += 1

        history.append(addr)
        cache_history.append(copy.copy(cache))

        for a in cache:
            cache_delta_occurences[a - addr] += 1

        if len(cache) < cache_size:
            cache.add(addr)
            misses_history.append(misses)
            continue

        if addr in cache:
            misses_history.append(misses)
        else:
            misses += 1

            cache_list = list(cache)
            cache_deltas = [(a - addr) % n_deltas for a in cache_list]
            history_deltas = [(a - addr) % n_deltas for a in history[-history_size:]]

            with torch.no_grad():
                model.eval()
                distr = model(
                    cache_list, cache_deltas, history[-history_size:], history_deltas
                )
            distr = distr[0]
            distr = distr.detach().cpu().numpy()

            if greedy:
                evict = cache_list[np.argmax(distr)]
            else:
                evict = random.choices(cache_list, weights=distr, k=1)[0]
            cache.remove(evict)

            cache.add(addr)

        if i > 100 and i % train_interval == 0 and train:
            samples = get_training_samples(
                cache_history,
                history,
                cache_size,
                history_size,
                n_elements,
                occurences=occ,
                n_samples=train_samples,
                n_deltas=n_deltas,
                sequence=sequence,
                cache_replacement=cache_replacement,
            )
            losses.append(train_model(samples, model, optimizer))
        misses_history.append(misses)
        pbar.set_description(f"Misses: {misses}, Loss: {losses[-1] if losses else 0}")

    print(in_cache)
    return misses, misses_history, losses, delta_occurences, cache_delta_occurences
