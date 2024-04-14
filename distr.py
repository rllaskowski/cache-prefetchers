import torch
import torch.nn as nn
from zmq import device

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

class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        super(SequenceModel, self).__init__()
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

# Example usage:
input_size = 100  # size of the vocabulary
hidden_size = 64  # size of the hidden state
output_size = 10  # number of elements in the output distribution
sequence_length = 4  # fixed length of the sequence

model = SequenceModel(input_size, hidden_size, output_size, sequence_length)
input_sequence = torch.tensor([[1, 2, 3, 4]])  # example input sequence
output_probabilities = model(input_sequence)
print(output_probabi
