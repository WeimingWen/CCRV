# -*- coding: utf-8 -*-

import random
import json
from nlp import Tokenizer


def preprocess():
    blog_en, blog_cn = [], []
    t = Tokenizer()
    print 'tokenize micro-blogs ...'
    with open('UM_Corpus/Bi-Microblog.txt', 'rU') as f:
        lines = f.readlines()
        for i in xrange(0, len(lines), 2):
            en = t.tokenize(lines[i].strip())
            cn = t.tokenize(lines[i + 1].strip(), cn=True)
            blog_en.append(en)
            blog_cn.append(cn)

    print 'save ...'
    with open('UM_Corpus/blog_en', 'w') as f1, open('UM_Corpus/blog_cn', 'w') as f2:
        json.dump(blog_en, f1)
        json.dump(blog_cn, f2)

    print 'tokenize news ...'
    news_en, news_cn = [], []
    with open('UM_Corpus/Bi-News.txt', 'rU') as f:
        lines = f.readlines()
        for i in xrange(0, len(lines), 2):
            en = t.tokenize(lines[i].strip())
            cn = t.tokenize(lines[i + 1].strip(), cn=True)
            news_en.append(en)
            news_cn.append(cn)

    print 'save ...'
    with open('UM_Corpus/news_en', 'w') as f3, open('UM_Corpus/news_cn', 'w') as f4:
        json.dump(news_en, f3)
        json.dump(news_cn, f4)

    t.close()

def load_dataset():
    """
    Load en and cn sentences
    """
    random.seed(777)
    with open('UM_Corpus/blog_en') as f1, open('UM_Corpus/blog_cn') as f2, open(
            'UM_Corpus/news_en') as f3, open('UM_Corpus/news_cn') as f4:
        blog_en = json.load(f1)
        blog_cn = json.load(f2)
        news_en = json.load(f3)
        news_cn = json.load(f4)

    # shuffle UM_Corpus
    blog = zip(blog_en, blog_cn)
    random.shuffle(blog)
    blog_en = [_[0] for _ in blog]
    blog_cn = [_[1] for _ in blog]
    news = zip(news_en, news_cn)
    random.shuffle(news)
    news_en = [_[0] for _ in news]
    news_cn = [_[1] for _ in news]

    train_en = blog_en[1000:] + news_en[1000:]
    train_cn = blog_cn[1000:] + news_cn[1000:]

    dev_en = blog_en[:1000] + news_en[:1000]
    dev_cn = blog_cn[:1000] + news_cn[:1000]

    return (train_en, train_cn), (dev_en, dev_cn)


if __name__ == '__main__':
    preprocess()
