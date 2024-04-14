from calendar import c
import torch
import torch.nn as nn
from zmq import device
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequenceModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
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


import torch
import torch.nn as nn

class ConstantSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * sequence_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.view(embedded.size(0), -1)  # Flatten the sequence
        hidden = torch.relu(self.fc1(embedded))
        output = self.fc2(hidden)
        output = self.softmax(output)
        return output


def get_training_samples(cache_history, access_history, cache_size, history_size, n_samples=20):
    samples = []

    while len(samples) < n_samples:
        i = random.randint(0, len(cache_history)-1)

        if len(cache_history[i]) < cache_size:
            continue

        cache = cache_history[i]
        history = access_history[i-history_size:i]
        

def train(samples, model, optimizer):
    x, y = zip(*samples)

    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    optimizer.zero_grad()

    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()

    return loss.item()



def test_on_sequence(sequence, cache_size, n_elements, train_interval=50):
    cache = set()
    losses = []
    history_size = 1
    model = ConstantSequenceModel(
        input_size=n_elements,
        hidden_size=64,
        output_size=cache_size,
        sequence_length=history_size+cache_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    history = []
    cache_history = []

    for i, item in enumerate(sequence):
        history.append(item)
        cache_history.append(cache)

        if len(cache) < cache_size:
            cache.add(item)
            continue

        if item in cache:
            continue

        distr = model(torch.tensor(history[-history_size:]+list(cache), dtype=torch.long).unsqueeze(0))
        distr = distr[0]
        distr = distr.detach().numpy()
        evict = random.choices(list(cache), weights=distr, k=1)[0]

        cache.remove(evict)

        cache.add(item)

        if i != 0 and i % train_interval == 0:
            samples = get_training_samples(cache_history, history, cache_size)
            train(samples, model, optimizer)
