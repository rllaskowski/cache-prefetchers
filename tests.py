from os import replace


from torch import cudnn_convolution_add_relu
from eviction import solve_linear_program
import random
from pprint import pprint
import functools
import itertools


def test_dom():
    ss = []
    ds = []
    for i in range(1000):
        N = 5
        pages = list(range(N))
        probs = {}
        for i in pages:
            for j in pages:
                if i == j:
                    continue
                probs[(i, j)] = random.random()
                probs[(j, i)] = 1 - probs[(i, j)]

        distr = solve_linear_program(probs, pages)
        assert distr is not None
        # pprint(probs)
        s = {}
        for i in pages:
            s[i] = sum(probs[(i, j)] for j in pages if j != i)
            # print(i, ':', s)
        ss.append(s[0])
        ds.append(distr[0])
        # print(distr)

    # calculate correlation
    print(np.corrcoef(ss, ds))


# test_dom()


def test_dom2():
    N = 50
    SEQ_LEN = 1000_00
    SAMPLES = 50000
    pages = range(N)

    probs = np.random.uniform(size=[N])
    probs = probs / sum(probs)

    sequence = [random.choices(pages, k=1, weights=probs)[0] for _ in range(SEQ_LEN)]
    not_before_occ = {(i, j): 0 for i, j in itertools.permutations(pages, 2)}

    for i in tqdm.tqdm(range(50000)):
        t = random.randint(0, SEQ_LEN)
        a, b = random.sample(pages, k=2)
        if a == b:
            continue

        for x in sequence[t:]:
            if x == a:
                not_before_occ[(b, a)] += 1

    prob_c = {}
    for a in pages:
        for b in pages:
            if a == b:
                continue
            s = not_before_occ[a, b] + not_before_occ[b, a]

            if s == 0:
                prob_c[(a, b)] = 0.5
            else:
                prob_c[(a, b)] = not_before_occ[(a, b)] / s

    cache_size = 8

    hits_dom = 0
    hits_random = 0
    better = 0
    worse = 0
    for i in tqdm.tqdm(range(SAMPLES)):
        cache = random.sample(pages, k=cache_size)

        probs_dom = solve_linear_program(prob_c, cache)
        t = random.randint(0, len(sequence))

        w = random.choices(cache, k=1, weights=probs_dom)[0]
        occ = set()
        a_ = 0
        for a in sequence[t:]:
            if a in cache and a != w:
                occ.add(a)
            if a == w:
                if len(occ) == len(cache) - 1:
                    hits_dom += 1
                    a_ = 1
                break

        w = random.choices(cache, k=1)[0]
        occ = set()
        b = 0
        for a in sequence[t:]:
            if a in cache and a != w:
                occ.add(a)
            if a == w:
                if len(occ) == len(cache) - 1:
                    hits_random += 1
                    b = 1
                break

        if a_ > b:
            better += 1

        if b > a_:
            worse += 1

    print("hits_dom", hits_dom, hits_dom / SAMPLES)
    print("hits_random", hits_random, hits_random / SAMPLES)
    print("better", better, better / SAMPLES)
    print("worse", worse, worse / SAMPLES)


test_dom2()
