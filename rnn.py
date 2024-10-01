from base64 import b16encode
from calendar import c
from copy import deepcopy
from math import e
from collections import defaultdict
import re
from IPython import embed
from matplotlib.pylab import f
from pyparsing import C, deque
from sympy import Q, field, im, li
import torch
from dataclasses import dataclass
import torch.nn as nn
import numpy as np
import random
import bisect
import itertools
import tqdm
import math
from mapping import Mapping
from utils import to_tensor_batched



class AddrHistory:
    def __init__(self, history_size: int, addr_mapping: Mapping) -> None:
        self.addrs = []
        self.pcs = []
        self.history_size = history_size
        self.addr_mapping = addr_mapping
        self.occurences = defaultdict(list)

    def update(self, addr: int, pc: int):
        self.addrs.append(addr)
        self.pcs.append(pc)
        self.occurences[addr].append(len(self.addrs) - 1)

    def get_addr_history(self, t, length=None):
        length = length or self.history_size
        return self.addrs[t + 1 - length : t + 1]

    def get_pc_history(self, t, length=None):
        length = length or self.history_size
        return self.pcs[t + 1 - self.history_size : t + 1]

    def __getitem__(self, t):
        return self.get_addr_history(t), self.get_pc_history(t)

    def sample(self, min_t=0, max_t=None):
        min_t = max(min_t, self.history_size)
        max_t = max_t or len(self.addrs)
        max_t = min(max_t, len(self.addrs))

        t = random.randint(min_t, max_t - 1)

        return self[t]



class CacheHistory:
    def __init__(self, max_size: int) -> None:
        self.history = []
        self.current = []
        self.max_size = max_size

    def add(self, addr: int, pc: int):
        if addr in self:
            self.current = [x for x in self.current if x[0] != addr]

        self.current.append((addr, pc))
        assert (
            len(self.current) <= self.max_size
        ), f"Cache size exceeded: {len(self.current)}"

    def evict(self, addr: int):
        assert addr in self, f"Address {addr} not in cache"
        previous_len = len(self.current)
        self.current = [x for x in self.current if x[0] != addr]
        assert len(self.current) == previous_len - 1, f"Eviction failed"

    def get_addr_history(self, t):
        return tuple([x[0] for x in self.history[t]])

    def get_pc_history(self, t):
        return tuple([x[1] for x in self.history[t]])

    def get_pc(self, addr):
        for a, pc in self.current:
            if a == addr:
                return pc

        raise ValueError(f"Address {addr} not in cache")

    def save(self):
        self.history.append(tuple(self.current))

    def __contains__(self, addr: int):
        return len([x for x in self.current if x[0] == addr]) > 0

    def __getitem__(self, t):
        return self.history[t]

    def __len__(self):
        return len(self.current)


def get_model_input(
    a: int,
    b: int,
    pc_a: int,
    pc_b: int,
    t: int,
    history: AddrHistory,
    delta_mapping: Mapping,
    addr_mapping: Mapping,
    pc_mapping: Mapping,
):
    history_t = history.get_addr_history(t)
    pc_history_t = history.get_pc_history(t)
    delta_history_t = history.get_deltas(t)

    return {
        "a": addr_mapping(a),
        "b": addr_mapping(b),
        "delta_a": delta_mapping(a - history_t[-1]),
        "delta_b": delta_mapping(b - history_t[-1]),
        "history": addr_mapping(history_t),
        "delta_history": delta_mapping(delta_history_t),
        "pc_history": pc_mapping(pc_history_t),
        "pc_a": pc_mapping(pc_a),
        "pc_b": pc_mapping(pc_b),
    }


def get_target_output(a: int, b: int, t: int, history: AddrHistory) -> float:
    a_first_occ = history.get_first_occurance(a, t + 1)
    b_first_occ = history.get_first_occurance(b, t + 1)

    if a_first_occ < b_first_occ:
        y = 1
    elif a_first_occ == b_first_occ:
        y = 0.5
    else:
        y = 0

    return y


class PCRnn(nn.Module):
    def __init__(
        self,
        pc_emb_size,
        hidden_size,
        pc_mapping: Mapping,
        dropout=0.1,
        rnn_layers=2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.concat_emb_size = pc_emb_size
        self.pc_emb_size = pc_emb_size
        self.pc_mapping = pc_mapping

        self.n_pcs = pc_mapping.max_size

        self.pc_embedding = nn.Embedding(self.n_pcs, pc_emb_size)

        self.rnn = nn.GRU(
            self.concat_emb_size, hidden_size, batch_first=True, dropout=dropout,
            num_layers=rnn_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.ff = nn.Linear(hidden_size, hidden_size*2)
        self.ff_pcs = nn.Linear(hidden_size*2, self.n_pcs)

        for layer in [self.ff_pcs, self.ff]:
            torch.nn.init.xavier_uniform_(layer.weight.data)

    def forward(
        self, pc_history
    ):
        pc_history = to_tensor_batched(pc_history, 2)
        history_emb = torch.cat(
            [
                self.pc_embedding(pc_history),
            ],
            dim=2,
        )

        if isinstance(self.rnn, nn.LSTM):
            _, (history_hn, __) = self.rnn(history_emb)
            history_hidden = history_hn.squeeze(0).unsqueeze(1)
        elif isinstance(self.rnn, nn.GRU):
            _, history_hn = self.rnn(history_emb)
            history_hidden = history_hn[-1].unsqueeze(1)
        else:
            raise ValueError("Invalid RNN type")


        out_dict = {}
        ff = history_hidden.squeeze(1)
        out = self.ff(ff)
        out = self.ff_pcs(out)
        out = torch.softmax(out, dim=1)
        out_dict["next_pc"] = out

        return out_dict


def to_batched_dict(list_dict):
    def _map(k):
        if k == "y":
            dtype = torch.float
        else:
            dtype = torch.long

        return torch.tensor([d[k] for d in list_dict], dtype=dtype)

    keys = list(list_dict[0].keys())
    return {k: _map(k) for k in keys}


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: list[tuple],
    batch_size=32,
):
    loss_pc_sum = 0
    criterion_pc = nn.CrossEntropyLoss()

    model.train()
    for i in range(0, len(samples), batch_size):
        batch = to_batched_dict(samples[i : i + batch_size])
        next_pc = batch.pop("next_pc")

        model_out = model(**batch)

        distr_pc = model_out["next_pc"]
        next_pc_target = torch.zeros_like(distr_pc)
        next_pc_target[range(len(next_pc)), next_pc] = 1
        loss_pc = criterion_pc(distr_pc, next_pc_target)
        loss = loss_pc
        loss_pc_sum += loss_pc.item()

        loss.backward()
        optimizer.step()

    batches = math.ceil(len(samples) / batch_size)
    loss_pc_sum /= batches

    return loss_pc_sum



def test_on_sequence(
    sequence_rw: list[tuple[int, str, int]],
    pc_mapping: Mapping,
    train_interval=50,
    history_size=1,
    emb_size=10,
    hidden_size=64,
    lr=1e-4,
    wd=0.1,
    dropout=0.1,
    train_samples=800,
    delta_emb_size=10,
    pc_emb_size=10,
    batch_size=32,
    rnn_layers=2,

):

    model_config = dict(
        history_size=history_size,
        emb_size=emb_size,
        delta_emb_size=delta_emb_size,
        pc_emb_size=pc_emb_size,
        hidden_size=hidden_size,
        dropout=dropout,
        pc_mapping=pc_mapping,
        rnn_layers=rnn_layers,
    )
    model = PCRnn(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


    samples = []
    current = deque(maxlen=history_size)
    for t, (a, rw, pc) in enumerate(sequence_rw):
        if len(current) < history_size:
            continue

        current.append(pc)
        samples.append(list(current))

        if len(samples) <= train_samples:
            samples.append(current)
