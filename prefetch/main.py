import argparse
import random

import alg.alg
import probmodel.ffm
import probmodel.probmodel
import probmodel.compression
import simulate
import traces.huawei
import traces.markov_chain


def gen_huawei_trace(trace_size, trace_path, cluster_size):
    trace = traces.huawei.parse_huawei_traces(trace_path)
    seq = traces.huawei.page_seq(trace, cluster_size)
    if trace_size > 0:
        seq = seq[:trace_size]
    return seq


def gen_markov_chain(n):
    mc = [
        ([0, 1, 2], [0.25, 0.5, 0.25]),
        ([2, 3], [0.5, 0.5]),
        ([1, 4], [0.75, 0.5]),
        ([0, 5], [0.1, 0.9]),
        ([5], [1.0]),
        ([0, 1, 2, 3, 4, 5], [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
    ]
    random.seed(0)
    return traces.markov_chain.markov_chain_seq(n, mc)


def test_opt(seq, cache_size):
    page_hist = dict()
    for t in range(0, len(seq)):
        p = seq[t]
        if p not in page_hist:
            page_hist[p] = [t]
        else:
            page_hist[p].append(t)
    for h in page_hist.values():
        h.reverse()
    cache = set()
    misses = 0
    for p in seq:
        page_hist[p].pop()
        if p not in cache:
            misses += 1
            if len(cache) == cache_size:
                best_t = 0
                best_q = None
                for q in cache:
                    h = page_hist[q]
                    if h:
                        t = h[-1]
                        if t > best_t:
                            best_t = t
                            best_q = q
                    else:
                        best_q = q
                        break
                cache.remove(best_q)
            cache.add(p)
    print(f'opt cache_size={cache_size:d} {misses:d}')


def test_dom_ffm(seq, cache_size, history_size, n, k, lambda_, eta, sps):
    ffm_model = probmodel.ffm.FFMSingle(n, history_size + 2, k, lambda_, eta, history_size, sps)
    algo = alg.alg.DomDistAlg(ffm_model)
    misses = simulate.simulate(cache_size, algo, seq)
    print(f'dom_ffm history_size={history_size:2d} n={n:3d} cache_size={cache_size:d} sps={sps:d} k={k:d}'
          f' lambda={lambda_:g} eta={eta:g} {misses:d}')


def test_dom_occurrences(seq, cache_size):
    model = probmodel.probmodel.ProportionalToOccurrences()
    algo = alg.alg.DomDistAlg(model)
    misses = simulate.simulate(cache_size, algo, seq)
    print(f'dom_occ cache_size={cache_size:d} {misses:d}')


def test_lru(seq, cache_size):
    algo = alg.alg.LRUAlg()
    misses = simulate.simulate(cache_size, algo, seq)
    print(f'lru cache_size={cache_size:d} {misses:d}')


def test_mq(seq, cache_size):
    algo = alg.alg.MQAlg()
    misses = simulate.simulate(cache_size, algo, seq)
    print(f'mq cache_size={cache_size:d} {misses:d}')


def test_met_occ(seq, cache_size, prefetch=0):
    model = probmodel.probmodel.ProportionalToOccurrences()
    algo = alg.alg.METAlg(model)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'met_occ_prefetch={prefetch:d} cache_size={cache_size:d} {misses:d}')


def test_lru_prefetch_next(seq, cache_size, prefetch):
    algo = alg.alg.LRUPrefetchMostProbable(probmodel.probmodel.PredictNext())
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'lru_prefetch_next cache_size={cache_size:d} {misses:d}')


def test_mq_prefetch_next(seq, cache_size, prefetch):
    algo = alg.alg.MQPrefetchMostProbable(probmodel.probmodel.PredictNext())
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'mq_prefetch_next cache_size={cache_size:d} {misses:d}')


def test_lru_prefetch_prop(seq, cache_size, prefetch):
    algo = alg.alg.LRUPrefetchMostProbable(probmodel.probmodel.ProportionalToOccurrences())
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'lru_prefetch_prop cache_size={cache_size:d} {misses:d}')


def test_lru_prefetch_lz(seq, cache_size, prefetch):
    algo = alg.alg.LRUPrefetchMostProbable(probmodel.compression.LZModel())
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'lru_prefetch_lz cache_size={cache_size:d} {misses:d}')


def test_lru_prefetch_ffm(seq, cache_size, prefetch, prob_threshold, history_size, n, k, lambda_, eta, sps):
    ffm_model = probmodel.ffm.FFMDistribution(n, history_size, k, lambda_, eta, history_size, sps)
    algo = alg.alg.LRUPrefetchMostProbable(ffm_model, prob_threshold)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'lru_prefetch_ffm history_size={history_size:2d} n={n:3d} cache_size={cache_size:d} sps={sps:d} k={k:d}'
          f' lambda={lambda_:g} eta={eta:g} prob_threshold={prob_threshold:g} {misses:d}')


def test_lru_prefetch_markov(seq, cache_size, prefetch, context_size):
    algo = alg.alg.LRUPrefetchMostProbable(probmodel.compression.MarkovModel(context_size))
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'lru_prefetch_markov cache_size={cache_size:d} {misses:d}')


def test_dom_ffm_prefetch_markov(seq, cache_size, prefetch, prob_threshold, context_size, history_size, n, k, lambda_, eta, sps):
    ffm_model = probmodel.ffm.FFMSingle(n, history_size + 2, k, lambda_, eta, history_size, sps)
    prefetch_model = probmodel.compression.MarkovModel(context_size)
    algo = alg.alg.DomDistAlg(ffm_model, prefetch_model, prob_threshold)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'dom_ffm_prefetch_markov history_size={history_size:2d} n={n:3d} cache_size={cache_size:d} sps={sps:d} k={k:d}'
          f' lambda={lambda_:g} eta={eta:g} prob_threshold={prob_threshold:g} {misses:d}')


def test_random(seq, cache_size):
    algo = alg.alg.RandomAlg()
    misses = simulate.simulate(cache_size, algo, seq)
    print(f'random cache_size={cache_size:d} {misses:d}')


def test_dom_ffm_prefetch_next(seq, cache_size, prefetch, history_size, n, k, lambda_, eta, sps):
    ffm_model = probmodel.ffm.FFMSingle(n, history_size + 2, k, lambda_, eta, history_size, sps)
    prefetch_model = probmodel.probmodel.PredictNext()
    algo = alg.alg.DomDistAlg(ffm_model, prefetch_model, 0.0)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'dom_ffm_prefetch_next cache_size={cache_size:d} {misses:d}')


def test_dom_ffm_prefetch_mixed(seq, cache_size, prefetch, prob_threshold, context_size, history_size, n, k, lambda_, eta, sps):
    ffm_model = probmodel.ffm.FFMSingle(n, history_size + 2, k, lambda_, eta, history_size, sps)
    markov_model = probmodel.compression.MarkovModel(context_size)
    next_model = probmodel.probmodel.PredictNext()
    prefetch_model = probmodel.probmodel.Mixed([(0.5, markov_model), (0.5, next_model)])
    algo = alg.alg.DomDistAlg(ffm_model, prefetch_model, prob_threshold)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'dom_ffm_prefetch_mixed history_size={history_size:2d} n={n:3d} cache_size={cache_size:d} sps={sps:d} k={k:d}'
          f' lambda={lambda_:g} eta={eta:g} prob_threshold={prob_threshold:g} {misses:d}')


def test_dom_lz(seq, cache_size, prefetch):
    pmodel = probmodel.compression.LZModel()
    algo = alg.alg.DomPrefetchMostProbable(pmodel)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'dom_lz_prefetch={prefetch:d} cache_size={cache_size:d} {misses:d}')


def test_dom_markov(seq, cache_size, prefetch, context_size=1):
    pmodel = probmodel.compression.MarkovModel(context_size)
    algo = alg.alg.DomPrefetchMostProbable(pmodel)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'dom_markov_prefetch={prefetch:d} cache_size={cache_size:d} {misses:d}')


def test_met_lz(seq, cache_size, prefetch=0):
    model = probmodel.compression.LZModel()
    algo = alg.alg.METAlg(model)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'met_lz_prefetch={prefetch:d} cache_size={cache_size:d} {misses:d}')


def test_met_markov(seq, cache_size, prefetch=0, context_size=3):
    model = probmodel.compression.MarkovModel(context_size)
    algo = alg.alg.METAlg(model)
    misses = simulate.simulate(cache_size, algo, seq, prefetch_size=prefetch)
    print(f'met_markov_prefetch={prefetch:d} cache_size={cache_size:d} {misses:d}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='dom_ffm')
    parser.add_argument('--cache_size', type=int, default=8)
    parser.add_argument('--history_size', type=int, default=6)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--lambda', type=float, default=1e-3, dest='lambda_')
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--samples_per_step', type=int, default=8)
    parser.add_argument('--prob_threshold', type=float, default=0.0)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--trace_path', type=str)
    parser.add_argument('--trace_size', type=int, default=0)
    parser.add_argument('--cluster_size', type=int, default=16384)
    parser.add_argument('--cyclic', type=int)
    parser.add_argument('--markov_chain', action='store_true')
    args = parser.parse_args()
    if args.cyclic is not None:
        seq = traces.markov_chain.cyclic_seq(args.trace_size, args.cyclic)
    elif args.markov_chain:
        seq = gen_markov_chain(args.trace_size)
    else:
        seq = gen_huawei_trace(args.trace_size, args.trace_path, args.cluster_size)
    match args.algo:
        case 'size':
            s = set()
            for p in seq:
                s.add(p)
            print(f'size length={len(seq):d} distinct={len(s):d}')
        case 'opt':
            test_opt(seq, args.cache_size)
        case 'random':
            test_random(seq, args.cache_size)
        case 'dom_ffm':
            test_dom_ffm(seq,
                         cache_size=args.cache_size,
                         history_size=args.history_size,
                         n=args.n,
                         k=args.k,
                         lambda_=args.lambda_,
                         eta=args.eta,
                         sps=args.samples_per_step)
        case 'lru':
            test_lru(seq, args.cache_size)
        case 'mq':
            test_mq(seq, args.cache_size)
        case 'lru_next_prefetch':
            test_lru_prefetch_next(seq, args.cache_size, args.prefetch)
        case 'mq_next_prefetch':
            test_mq_prefetch_next(seq, args.cache_size, args.prefetch)
        case 'lru_occ_prefetch':
            test_lru_prefetch_prop(seq, args.cache_size, args.prefetch)
        case 'lru_lz_prefetch':
            test_lru_prefetch_lz(seq, args.cache_size, args.prefetch)
        case 'lru_markov_prefetch':
            test_lru_prefetch_markov(seq, args.cache_size, args.prefetch, context_size=3)
        case 'dom_ffm_next_prefetch':
            test_dom_ffm_prefetch_next(seq,
                                       cache_size=args.cache_size,
                                       prefetch=args.prefetch,
                                       history_size=args.history_size,
                                       n=args.n,
                                       k=args.k,
                                       lambda_=args.lambda_,
                                       eta=args.eta,
                                       sps=args.samples_per_step)
        case 'dom_ffm_markov_prefetch':
            test_dom_ffm_prefetch_markov(seq,
                                         cache_size=args.cache_size,
                                         prefetch=args.prefetch,
                                         prob_threshold=args.prob_threshold,
                                         context_size=3,
                                         history_size=args.history_size,
                                         n=args.n,
                                         k=args.k,
                                         lambda_=args.lambda_,
                                         eta=args.eta,
                                         sps=args.samples_per_step)
        case 'dom_ffm_mixed_prefetch':
            test_dom_ffm_prefetch_mixed(seq,
                                        cache_size=args.cache_size,
                                        prefetch=args.prefetch,
                                        prob_threshold=args.prob_threshold,
                                        context_size=3,
                                        history_size=args.history_size,
                                        n=args.n,
                                        k=args.k,
                                        lambda_=args.lambda_,
                                        eta=args.eta,
                                        sps=args.samples_per_step)
        case 'lru_ffm_prefetch':
            test_lru_prefetch_ffm(seq,
                                  cache_size=args.cache_size,
                                  prefetch=args.prefetch,
                                  prob_threshold=args.prob_threshold,
                                  history_size=args.history_size,
                                  n=args.n,
                                  k=args.k,
                                  lambda_=args.lambda_,
                                  eta=args.eta,
                                  sps=args.samples_per_step)
        case 'dom_lz_prefetch':
            test_dom_lz(seq, args.cache_size, args.prefetch)
        case 'dom_markov_prefetch':
            test_dom_markov(seq, args.cache_size, args.prefetch, context_size=3)
        case 'dom_lz':
            test_dom_lz(seq, args.cache_size, 0)
        case 'dom_occ':
            test_dom_occurrences(seq, args.cache_size)
        case 'dom_markov':
            test_dom_markov(seq, args.cache_size, 0, context_size=3)
        case 'met_occ':
            test_met_occ(seq, args.cache_size)
        case 'met_occ_prefetch':
            test_met_occ(seq, args.cache_size, args.prefetch)
        case 'met_lz':
            test_met_lz(seq, args.cache_size)
        case 'met_lz_prefetch':
            test_met_lz(seq, args.cache_size, args.prefetch)
        case 'met_markov':
            test_met_markov(seq, args.cache_size)
        case 'met_markov_prefetch':
            test_met_markov(seq, args.cache_size, args.prefetch)
        case _:
            print('Unknown algorithm name', args.algo)


if __name__ == '__main__':
    main()

# mathieu is here
