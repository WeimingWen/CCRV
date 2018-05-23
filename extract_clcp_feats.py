import json
import numpy as np
from utils import extract_feats, task_split, event_split
from tools import imputer, standardize
from scipy.stats import pearsonr
import cPickle as pkl


def exp1():
    print 'extract features for experiment 1'
    with open('CCMR/CCMR_Twitter_t.txt') as f1, open('CCMR/CCMR_Google_t.txt') as f2:
        twitter = json.load(f1)
        google = json.load(f2)

    # get split of train test
    cv_task = task_split(twitter)
    cv_event = event_split(twitter)

    print 'extract cpcl features'
    X, Y = extract_feats(twitter, google, embed='complete', agree='complete')

    X_fill = imputer(X)
    X_fill = standardize(X_fill)
    scores = []
    for i in xrange(10):
        p = pearsonr(X_fill[:, i], Y)
        print p
        scores.append(abs(p[0]))
    print 'average: ', np.average(scores)

    with open('CLCP/Twitter_CLCP_via_Google.pkl','wb') as f:
        pkl.dump(((X,Y),(cv_task, cv_event)), f)


def exp2():
    print 'extract features for experiment 2'
    with open('CCMR/CCMR_Twitter_t.txt') as f1, open('CCMR/CCMR_Baidu_t.txt') as f2:
        twitter = json.load(f1)
        baidu = json.load(f2)

    # get split of train test
    cv_task = task_split(twitter)
    cv_event = event_split(twitter)

    print 'extract cpcl features'
    X, Y = extract_feats(twitter, baidu)

    X_fill = imputer(X)
    X_fill = standardize(X_fill)
    scores = []
    for i in xrange(10):
        p = pearsonr(X_fill[:, i], Y)
        print p
        scores.append(abs(p[0]))
    print 'average: ', np.average(scores)

    with open('CLCP/Twitter_CLCP_via_Baidu.pkl', 'wb') as f:
        pkl.dump(((X, Y), (cv_task, cv_event)), f)

def exp2_combo():
    print 'extract features for experiment 1'
    with open('CCMR/CCMR_Twitter_t.txt') as f1, open('CCMR/CCMR_Google_t.txt') as f2:
        twitter = json.load(f1)
        google = json.load(f2)
    with open('CCMR/CCMR_Baidu_t.txt') as f3:
        baidu=json.load(f3)
    # get split of train test
    cv_task = task_split(twitter)
    cv_event = event_split(twitter)

    print 'extract cpcl features'
    X, Y = extract_feats(twitter, google+baidu, embed='complete', agree='complete')

    X_fill = imputer(X)
    X_fill = standardize(X_fill)
    scores = []
    for i in xrange(10):
        p = pearsonr(X_fill[:, i], Y)
        print p
        scores.append(abs(p[0]))
    print 'average: ', np.average(scores)

    with open('CLCP/Twitter_CLCP_via_All.pkl','wb') as f:
        pkl.dump(((X,Y),(cv_task, cv_event)), f)


def exp3():
    print 'extract features for experiment 3'
    with open('CCMR/CCMR_Baidu_t.txt') as f1, open('CCMR/CCMR_Google_t.txt') as f2:
        baidu = json.load(f1)
        google = json.load(f2)

    # filter others rumors in Baidu
    baidu_p = []
    for elem in baidu:
        if elem['label'] in [0, 1]:
            baidu_p.append(elem)

    # get split of train test
    cv_task = task_split(baidu_p)
    cv_event = event_split(baidu_p)

    print 'extract cpcl features'
    X, Y = extract_feats(baidu_p, google)

    with open('CLCP/Baidu_CLCP_via_Google.pkl', 'wb') as f:
        pkl.dump(((X, Y), (cv_task, cv_event)), f)



if __name__ == '__main__':
    exp1()
    exp2()
    exp2_combo()
    exp3()