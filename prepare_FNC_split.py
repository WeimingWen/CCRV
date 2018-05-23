import csv
import numpy as np
from multilingual_embedding_module import Embedder
from nlp import Tokenizer
import torch
import cPickle as pkl
import random

stances = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}


class News(object):
    # object for processing and presenting news to clf

    def __init__(self, stances='FNC_1/train_stances.csv', bodies='FNC_1/train_bodies.csv'):
        # process files into arrays, etc
        self.bodies = self.proc_bodies(bodies)
        self.headlines = []
        with open(stances, 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for line in reader:
                if len(line) == 2:
                    hl, bid = line
                    stance = 'unrelated'
                else:
                    hl, bid, stance = line
                self.headlines.append((hl, bid, stance))
        self.n_headlines = len(self.headlines)

    def get_one(self, ridx=None):
        # select a single sample either randomly or by index
        if ridx is None:
            ridx = np.random.randint(0, self.n_headlines)
        head = self.headlines[ridx]
        body = self.bodies[head[1]]

        return head, body

    def sample(self, n=16, ridx=None):
        # select a batch of samples either randomly or by index
        heads = []
        bodies = []
        stances_d = []
        if ridx is not None:
            for r in ridx:
                head, body_text = self.get_one(r)
                head_text, _, stance = head
                heads.append(head_text)
                bodies.append(body_text)
                stances_d.append(stances[stance])
        else:
            for i in xrange(n):
                head, body_text = self.get_one()
                head_text, _, stance = head
                heads.append(head_text)
                bodies.append(body_text)
                stances_d.append(stances[stance])

        return heads, bodies, stances_d

    def validate(self):
        # iterate over the dataset in order
        for i in xrange(len(self.headlines)):
            yield self.sample(ridx=[i])

    def proc_bodies(self, fn):
        # process the bodies csv into arrays
        tmp = {}
        with open(fn, 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for line in reader:
                bid, text = line
                tmp[bid] = text
        return tmp


def encode_stance(s):
    code = [0, 0, 0, 0]
    code[s] = 1
    return code


def prepare_data():
    embed = 'complete'
    batch_size = 32

    # FNC Baseline training set
    news = News(stances='FNC_1/train.csv', bodies='FNC_1/train_bodies.csv')
    # FNC baseline validation set
    val_news = News(stances='FNC_1/test.csv', bodies='FNC_1/train_bodies.csv')
    # FNC test set
    test_news = News(stances='FNC_1/competition_test_stances.csv',
                     bodies='FNC_1/competition_test_bodies.csv')
    print news.n_headlines, val_news.n_headlines, test_news.n_headlines

    t = Tokenizer()

    # format training CCMR
    h, b, s = news.sample(ridx=range(news.n_headlines))
    hv, bv, sv = val_news.sample(ridx=range(val_news.n_headlines))
    ht, bt, st = test_news.sample(ridx=range(test_news.n_headlines))

    data = zip(h + hv + ht, b + bv + bt, s + sv + st)

    # choose eval data
    random.seed(777)
    random.shuffle(data)

    train, dev = [], []
    n0, n1, n2, n3 = 0, 0, 0, 0
    for head, body, stance in data:
        head = t.tokenize(head)
        body = t.tokenize(body)

        if stance == 0:
            if n0 >= 250:
                n0 += 1
                train.append((head, body, stance))
            else:
                n0 += 1
                dev.append((head, body, stance))
        elif stance == 1:
            if n1 >= 250:
                n1 += 1
                train.append((head, body, stance))
            else:
                n1 += 1
                dev.append((head, body, stance))
        elif stance == 2:
            if n2 >= 250:
                n2 += 1
                train.append((head, body, stance))
            else:
                n2 += 1
                dev.append((head, body, stance))
        elif stance == 3:
            if n3 >= 250:
                n3 += 1
                train.append((head, body, stance))
            else:
                n3 += 1
                dev.append((head, body, stance))
        else:
            print 'error!'

    print 'stastistics: ', n0, n1, n2, n3

    emb = Embedder(embed)
    print 'start embedding ...'
    train_tensor = []
    for ibx in xrange(0, len(train), batch_size):
        if ibx + batch_size >= len(train):
            batch = train[ibx:len(train)]
        else:
            batch = train[ibx:ibx + batch_size]
        head, body = emb.embed([_[0] for _ in batch], [_[1] for _ in batch])

        # head = torch.FloatTensor(head)
        # body = torch.FloatTensor(body)
        # stance = torch.FloatTensor([encode_stance(_[2]) for _ in batch])
        stance = [_[2] for _ in batch]
        train_tensor.append((head, body, stance))

    vh, vb = emb.embed([_[0] for _ in dev], [_[1] for _ in dev])
    vh = torch.FloatTensor(vh)
    vb = torch.FloatTensor(vb)
    vs = [_[2] for _ in dev]

    print 'save ...'
    with open('FNC_1/train.pkl', 'wb') as f:
        pkl.dump(train_tensor, f)

    with open('FNC_1/dev.pkl', 'wb') as f:
        pkl.dump((vh,vb,vs), f)


if __name__ == '__main__':
    prepare_data()