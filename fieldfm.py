import numpy as np
from sympy import field
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
import bisect

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype='int64')
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x.to(device)
        assert isinstance(x, torch.Tensor), f"emb layer: {type(x)}"
        assert x.device == device, f"emb layer: {x.device} != {device}"

        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class _FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype='int64')
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x.to(device)
        assert isinstance(x, torch.Tensor), f"_ffm: {type(x)}"
        assert x.device == device, f"_ffm: {x.device} != {device}"

        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype='int64')

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)
        x = x.to(device)

        assert x.device == device, f"linear: {x.device} != {device}"
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FieldAwareFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.

    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = _FieldAwareFactorizationMachine(field_dims, embed_dim)
        self.n = field_dims[0]

        self.to(device)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)

        x = x.to(device)

        x = x % self.n
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))



class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, n, m, k):
        nn.Embedding
        super().__init__()
        self.n = n
        self.m = m
        self.k = k
        self.field_embeddings = nn.Parameter(torch.randn(n, m, k), requires_grad=True)
        self.device = device
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)

        x.to(self.device)
        x = x % self.n

        if len(x.shape) == 1:
            x.unsqueeze_(0)

        embeddings = self.field_embeddings[x]
        i1, i2 = torch.triu_indices(self.m, self.m, offset=1)

        embeds1 = embeddings[:, i1, i2, :]
        embeds2 = embeddings[:, i2, i1, :]

        interactions = (embeds1 * embeds2).sum(dim=-1).sum(dim=-1)

        probability = torch.sigmoid(interactions)

        return probability


def sample_data_point(
    access_history,
    cache_history: list[set],
    h: int,
    samples=4,
    fair=False,
    occurences: Optional[Dict[int, List[int]]] = None,
):
    """
    history: List of page requests (s0, s1, ..., sT-1)
    cache: Current set of pages in the cache
    h: History length
    Returns a sample data point for training
    """
    T = len(access_history)
    sample = []

    for _ in range(samples):
        if not fair or T - 2 - 50 < 0:
            t = np.random.randint(h, T - 2)
        else:
            t = np.random.choice(
                

        cache = cache_history[t]
        if len(cache) < 2:
            continue

        a, b = np.random.choice(list(cache), 2, replace=False)

        if occurences:
            if a not in occurences:
                r_a = float("inf")
            else:
                r_a = bisect.bisect_left(occurences[a], t)
                if r_a == len(occurences[a]):
                    r_a = float("inf")
            if b not in occurences:
                r_b = float("inf")
            else:
                r_b = bisect.bisect_left(occurences[b], t)
                if r_b == len(occurences[b]):
                    r_b = float("inf")
        else:
            raise ValueError("occurences is required")
            r_a = next((t_prime for t_prime in range(t, T) if access_history[t_prime] == a), float("inf"))
            r_b = next((t_prime for t_prime in range(t, T) if access_history[t_prime] == b), float("inf"))

        if r_a < r_b:
            y = 1
        elif r_a == r_b:
            y = 0.5
        else:
            y = 0

        sample.append((y, [a, b] + access_history[t - h : t]))

    return sample


def train_ffm(
    ffm,
    optimizer,
    history,
    cache_history: list[set],
    h,
    epochs,
    epoch_samples=4,
    occurences: Optional[Dict[int, List[int]]] = None,
):
    ffm.train()
    if len(history) <= h + 2:
        return []


    criterion = nn.BCELoss()
    losses = []

    for _ in range(epochs):
        data_point = sample_data_point(history, cache_history, h, epoch_samples, occurences=occurences)

        x_tensor = torch.tensor([data[1] for data in data_point], dtype=torch.long, device=device)
        y_tensor = torch.tensor(
            [data[0] for data in data_point], dtype=torch.float32, device=device
        )

        optimizer.zero_grad()
        outputs = ffm(x_tensor)

        loss = criterion(outputs, y_tensor)

        loss.backward()
        losses.append(loss.item())

        optimizer.step()

    return losses
