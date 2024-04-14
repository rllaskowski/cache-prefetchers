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
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype="int64")
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
        self.embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(sum(field_dims), embed_dim)
                for _ in range(self.num_fields)
            ]
        )
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype="int64")
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
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype="int64")

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
