from collections import defaultdict
from crosslingual_crossplatform_features import CLCP
import numpy as np
import cPickle as pkl


def extract_feats(target, data, base=False, embed='complete', agree='complete'):
    if base:
        with open('resources/add_dict.txt') as f:
            add_dict=pkl.load(f)
    # find webpages' titles with the same multi-modal content
    clcp = CLCP(embed=embed, agree=agree)
    related_sources = defaultdict(list)
    for elem in data:
        for id in elem['image_id']:
            related_sources[id].append(elem['title'])
    X, Y = [], []
    for elem in target:
        titles = []
        for id in elem['image_id']:
            titles += related_sources[id]
        try:
            content = elem['content']
        except:
            content = elem['title']
        feats = clcp.extract(content, titles)
        if base:
            feats=np.concatenate([feats, np.array(add_dict[elem['tweet_id']], float)])
        X.append(feats)
        Y.append(elem['label'])
    X = np.stack(X, axis=0)
    Y = np.array(Y)
    return X, Y


def event_split(data):
    events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback', 'columbianChemicals', 'passport', 'elephant',
              'underwater', 'livr', 'pigFish', 'eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']
    tests = [[] for _ in xrange(len(events))]
    for i in xrange(len(data)):
        tests[events.index(data[i]['event'])].append(i)
    trains = []
    for i in xrange(len(tests)):
        train = []
        for j in xrange(len(tests)):
            if j == i:
                continue
            train += tests[j]
        trains.append(train)
    cv = zip(np.array(trains), np.array(tests))

    return cv


def task_split(data):
    test, train = [], []
    for i in xrange(len(data)):
        if data[i]['event'] in ['eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']:
            test.append(i)
        else:
            train.append(i)

    cv = zip(np.array([train]), np.array([test]))

    return cv
