import math
import sys
from collections import defaultdict

import numpy

from probmodel.probmodel import ProbModel
from traces.mem_hist import MemHist


class TrieNode:

    def __init__(self, weight=0):
        self.weight = weight
        self.children = dict()

    # for debug
    def print(self, dep=0, edge=None):
        for i in range(0, dep):
            print(' ', file=sys.stderr, end='')
        if edge is None:
            print("root, w: %d" % self.weight, file=sys.stderr)
        else:
            print("%d, w: %d" % (edge, self.weight), file=sys.stderr)
        for k, v in self.children.items():
            v.print(dep + 1, k)


class LZModel(ProbModel):
    def __init__(self):
        self.trie_root = TrieNode()
        self.current_node = self.trie_root
        self.pages_seen = set()
        self.current_path = [self.trie_root]

    # update the model with the most recent history
    # supposed to be called after every change to the history
    def update(self, history: MemHist):
        last_page = next(history.get_history_at(0))
        self.pages_seen.add(last_page)
        if last_page in self.current_node.children:
            self.current_node = self.current_node.children[last_page]
            self.current_path.append(self.current_node)
        else:
            self.current_node.children[last_page] = TrieNode(1)
            for v in self.current_path:
                v.weight += 1
            self.current_node = self.trie_root
            self.current_path = [self.trie_root]

    # prob that page will appear in the next request
    def prob_next(self, page):
        # if current state is a leaf, look at the root
        node = self.current_node if len(self.current_node.children) > 0 else self.trie_root
        if page in node.children:
            return node.children[page].weight / node.weight
        else:
            return 0

    # prob that page_a will be requested before page_b
    def prob_before(self, page_a, page_b):
        if page_a == page_b:
            return 0
        if page_a in self.pages_seen:
            if page_b not in self.pages_seen:
                return 1
        else:
            if page_b not in self.pages_seen:
                return 0.5
            else:
                return 0

        def compute_root(node):
            mt, at = node.weight, 0
            for page, child in node.children.items():
                mt -= child.weight
                if page == page_a:
                    at += child.weight
                elif page != page_b:
                    (m, a) = compute_root(child)
                    mt += child.weight * m
                    at += child.weight * a
            mt /= node.weight
            at /= node.weight
            return mt, at

        mul, add = compute_root(self.trie_root)
        root_prob = add / (1 - mul)

        def compute_any(node):
            result = 0
            root_weight = node.weight
            for page, child in node.children.items():
                root_weight -= child.weight
                if page == page_a:
                    result += child.weight
                elif page != page_b:
                    result += child.weight * compute_any(child)
            result = (result + root_weight * root_prob) / node.weight
            return result

        return compute_any(self.current_node)

    # 1-based expected time when page is requested
    def expected_next(self, page):
        if page not in self.pages_seen:
            return math.inf

        def compute_root(node):
            mt, at = node.weight, node.weight
            for pg, child in node.children.items():
                mt -= child.weight
                if pg != page:
                    (m, a) = compute_root(child)
                    mt += child.weight * m
                    at += child.weight * a
            mt /= node.weight
            at /= node.weight
            return mt, at

        mul, add = compute_root(self.trie_root)
        root_expected = add / (1 - mul)

        def compute_any(node):
            root_weight = node.weight
            result = node.weight
            for pg, child in node.children.items():
                root_weight -= child.weight
                if pg != page:
                    result += child.weight * compute_any(child)
            return (result + root_weight * root_expected) / node.weight

        return compute_any(self.current_node)

    # return a dict known_pages -> prob_next
    def distr_next(self):
        # if current state is a leaf, look at the root
        node = self.current_node if len(self.current_node.children) > 0 else self.trie_root
        return {k: (v.weight / node.weight) for k, v in node.children.items()}

    def known_pages(self):
        return self.pages_seen


# Counting contexts of a given size; the "next" distribution follows the frequency of possible
# historical continuations
class MarkovModel(ProbModel):

    # todo limit history size if needed
    def __init__(self, k, history_size=math.inf):
        self.k = k
        self.pages_seen = set()
        self.history_size = history_size
        self.context_weight = defaultdict(int)
        self.context_edges = defaultdict(lambda: defaultdict(int))
        self.current_context = ()
        self.context_weight[()] = 1

    def _extend_context(self, context, page):
        return (context[1:] if len(context) == self.k else context) + (page,)

    def update(self, history: MemHist):
        last_page = next(history.get_history_at(0))
        self.pages_seen.add(last_page)
        for i in range(0, len(self.current_context) + 1):
            self.context_edges[self.current_context[i:]][last_page] += 1
        self.current_context = self._extend_context(self.current_context, last_page)
        for i in range(0, len(self.current_context) + 1):
            self.context_weight[self.current_context[i:]] += 1

    def prob_next(self, page):
        for i in range(0, self.k + 1):
            context = self.current_context[i:]
            if context in self.context_edges:
                if page in self.context_edges[context]:
                    return self.context_edges[context][page] / (self.context_weight[context] - 1)
                else:
                    return 0
        return 0

    def distr_next(self):
        for i in range(0, self.k + 1):
            context = self.current_context[i:]
            if context in self.context_edges:
                return {p: self.context_edges[context][p] / (self.context_weight[context] - 1)
                        for p in self.context_edges[context]}

    def _map_to_ints(self):
        mapping = dict()
        i = 0
        for k in self.context_weight.keys():
            mapping[k] = i
            i += 1
        return mapping

    def expected_next(self, page):
        n = len(self.context_weight)
        if n == 1:
            return math.inf
        idx = self._map_to_ints()
        a = numpy.zeros((n, n))
        b = numpy.ones((n, 1))
        for c, w in self.context_weight.items():
            cid = idx[c]
            a[cid][cid] = 1
            if c == () or c == self.current_context[-len(c):]:
                total_weight = w - 1
            else:
                total_weight = w
            if c in self.context_edges:
                for next_page, count in self.context_edges[c].items():
                    if next_page != page:
                        cp = self._extend_context(c, next_page)
                        a[cid][idx[cp]] -= count / total_weight
            else:
                b[cid][0] = 0
                a[cid][idx[c[1:]]] = -1
        try:
            sol = numpy.linalg.solve(a, b)
            return sol[idx[self.current_context]][0]
        except numpy.linalg.LinAlgError:
            return math.inf

    def prob_before(self, page_a, page_b):
        if page_a == page_b:
            return 0
        if page_a in self.pages_seen:
            if page_b not in self.pages_seen:
                return 1
        else:
            if page_b not in self.pages_seen:
                return 0.5
            else:
                return 0
        n = len(self.context_weight)
        idx = self._map_to_ints()
        a = numpy.zeros((n, n))
        b = numpy.zeros((n, 1))
        for c, w in self.context_weight.items():
            cid = idx[c]
            a[cid][cid] = 1
            if c == () or c == self.current_context[-len(c):]:
                total_weight = w - 1
            else:
                total_weight = w
            if c in self.context_edges:
                for next_page, count in self.context_edges[c].items():
                    if next_page == page_a:
                        b[cid][0] += count / total_weight
                    elif next_page != page_b:
                        cp = self._extend_context(c, next_page)
                        a[cid][idx[cp]] -= count / total_weight
            else:
                a[cid][idx[c[1:]]] = -1
        try:
            sol = numpy.linalg.solve(a, b)
            return sol[idx[self.current_context]][0]
        except numpy.linalg.LinAlgError:
            return math.inf

    def known_pages(self):
        return self.pages_seen
