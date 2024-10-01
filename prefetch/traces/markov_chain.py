import random


def markov_chain_seq(n, markov_chain):
    seq = [0]
    while len(seq) < n:
        state = seq[-1]
        transitions = markov_chain[state]
        seq.append(random.choices(transitions[0], weights=transitions[1])[0])
    return seq


def cyclic_seq(n, cycle_size):
    return [i % cycle_size for i in range(n)]
