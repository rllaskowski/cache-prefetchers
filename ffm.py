import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, n, m, k):
        super().__init__()
        self.n = n
        self.m = m
        self.k = k
        self.field_embeddings = nn.Parameter(torch.randn(n, m, k), requires_grad=True)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)

        if len(x.shape) == 1:
            x.unsqueeze_(0)

        embeddings = self.field_embeddings[x]
        interactions = torch.zeros(x.shape[0])
        for j1 in range(self.m):
            for j2 in range(j1 + 1, self.m):
                interactions = interactions + (embeddings[:, j1, j2, :] * embeddings[:, j2, j1, :]).sum(dim=1)

        probability = torch.sigmoid(interactions)

        return probability


def sample_data_point(history, cache: set, h: int, samples=4):
    """
    history: List of page requests (s0, s1, ..., sT-1)
    cache: Current set of pages in the cache
    h: History length
    Returns a sample data point for training
    """
    T = len(history)
    t = np.random.randint(h, T-2)
    sample = []

    for _ in range(samples):
        a, b = np.random.choice(list(cache), 2, replace=False)

        r_a = next((t_prime for t_prime in range(t, T) if history[t_prime] == a), float('inf'))
        r_b = next((t_prime for t_prime in range(t, T) if history[t_prime] == b), float('inf'))

        if r_a < r_b:
            y = 1
        elif r_a == r_b:
            y = 0.5
        else:
            y = 0

        sample.append((y, [a,b] + history[t-h:t]))


    return sample


def train_ffm(ffm, history, cache: set, h, epochs, lr, wd=0.1, epoch_samples=4):
    if len(history) <= h+2:
        return []

    optimizer = optim.AdamW(ffm.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCELoss()
    losses = []

    for _ in range(epochs):
        data_point = sample_data_point(history, cache, h, epoch_samples)

        x_tensor = torch.tensor([data[1] for data in data_point], dtype=torch.long)
        y_tensor = torch.tensor([data[0] for data in data_point], dtype=torch.float32)

        optimizer.zero_grad()
        outputs = ffm(x_tensor)

        loss = criterion(outputs, y_tensor)

        loss.backward()
        losses.append(loss.item())

        optimizer.step()

    return losses
