# -*- coding: utf-8 -*-

import numpy as np
from numpy import einsum
from multilingual_embedding_module import Embedder
from agreement_classifier import Stancer

# Cross-lingual Cross-platform Features
class CLCP:
    def __init__(self, embed='submit', agree='submit'):
        self.embedder = Embedder(embed)
        self.clf = Stancer(agree)

    def extract(self, content, titles):
        if not titles:
            return np.array([np.nan for _ in xrange(10)])
        vec_c, vec_t = self.embedder.embed([content for _ in xrange(len(titles))], titles)
        distance = np.ones(len(titles))+np.dot(vec_c[0],vec_t.T)

        agreement = self.clf.compute_stance(head=vec_t, body=vec_c)

        return np.concatenate([self.compute_dist(distance), self.compute_agree(agreement)])

    def compute_dist(self, distance):
        # dist feats
        mean = np.average(distance)
        variance = np.var(distance)
        return [mean, variance]

    def compute_agree(self, agreement):
        # agree feats
        means = np.average(agreement, axis=0)
        variance = np.var(agreement, axis=0)
        return [means[0], variance[0], means[1], variance[1], means[2], variance[2], means[3], variance[3]]
