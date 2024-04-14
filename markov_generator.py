import random

import numpy as np


class RandomMarkovGenerator:
    def __init__(self, num_states):
        self.num_states = num_states
        self.transition_matrix = self.generate_random_transition_matrix()

    def generate_random_transition_matrix(self):
        matrix = np.random.rand(self.num_states, self.num_states)
        matrix /= matrix.sum(axis=1)[:, None]

        return matrix

    def generate_sequence(self, length):
        sequence = [random.randint(0, self.num_states - 1)]

        for _ in range(length - 1):
            current_state = sequence[-1]
            next_state = np.random.choice(
                self.num_states, p=self.transition_matrix[current_state]
            )
            sequence.append(next_state)

        return sequence
