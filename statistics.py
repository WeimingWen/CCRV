import json
from collections import defaultdict

events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback', 'columbianChemicals', 'passport', 'elephant',
          'underwater', 'livr', 'pigFish', 'eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']


def Twitter():
    with open('CCMR/CCMR_Twitter.txt') as f:
        data = json.load(f)

    for event in events:
        print event
        fake = 0
        real = 0
        for tweet in data:
            if tweet['event'] == event:
                if tweet['label'] == 1:
                    fake += 1
                else:
                    real += 1

        print 'num ', fake + real
        print 'fake', fake
        print 'real', real


def build_google():
    with open('CCMR/google_webpages.txt') as f:
        data = json.load(f)

    dict = {}
    for id in data:
        for source in data[id][:20]:
            if source['url'] not in dict:
                elem={}
                elem['image_id']=[id]
                elem['title']=source['title']
                elem['label']=source['annotation']
                elem['url']=source['url']
                elem['event']=source['event']
                dict[source['url']]=elem
            else:
                if id not in dict[source['url']]['image_id']:
                    dict[source['url']]['image_id'].append(id)

    # save
    dataset=[]
    for url in dict:
        dataset.append(dict[url])
    with open('CCMR/CCMR_Google.txt','w') as f:
        json.dump(dataset, f)


def build_baidu():
    with open('CCMR/baidu_webpages.txt') as f:
        data = json.load(f)

    dict = {}
    for id in data:
        for source in data[id][:20]:
            if source['url'] not in dict:
                elem={}
                elem['image_id']=[id]
                elem['title']=source['title']
                elem['label']=source['annotation']
                elem['url']=source['url']
                elem['event'] = source['event']
                dict[source['url']]=elem
            else:
                if id not in dict[source['url']]['image_id']:
                    dict[source['url']]['image_id'].append(id)

    # save
    dataset=[]
    for url in dict:
        dataset.append(dict[url])
    with open('CCMR/CCMR_Baidu.txt','w') as f:
        json.dump(dataset, f)

def stas():
    with open('CCMR/CCMR_Baidu.txt') as f:
        data = json.load(f)

    for event in events:
        print event
        fake = 0
        real = 0
        others = 0
        for tweet in data:
            if tweet['event'] == event:
                if tweet['label'] == 1:
                    fake += 1
                elif tweet['label'] == 0:
                    real += 1
                elif tweet['label'] == 2:
                    others += 1
                else:
                    print 'error!'
                    print tweet


        print 'num ', fake + real+ others
        print 'fake', fake
        print 'real', real
        print 'others', others

    print 'summary'
    n,f,r,o=0,0,0,0
    for tweet in data:
        n+=1
        if tweet['label'] == 1:
            f += 1
        elif tweet['label'] == 0:
            r += 1
        elif tweet['label'] == 2:
            o += 1
        else:
            print 'error!'
            print tweet

    print 'number',n,'fake',f,'real',r,'others',o

def fnc():
    from prepare_FNC_split import News
    # FNC Baseline training set
    news = News(stances='FNC_1/train.csv', bodies='FNC_1/train_bodies.csv')
    # FNC baseline validation set
    val_news = News(stances='FNC_1/test.csv', bodies='FNC_1/train_bodies.csv')

    test_news = News(stances='FNC_1/competition_test_stances.csv',
                     bodies='FNC_1/competition_test_bodies.csv')
    print news.n_headlines, val_news.n_headlines, test_news.n_headlines

    # format training CCMR
    h, b, s = news.sample(ridx=range(news.n_headlines))
    hv, bv, sv = val_news.sample(ridx=range(val_news.n_headlines))
    ht, bt, st = test_news.sample(ridx=range(test_news.n_headlines))

    data = zip(h + hv + ht, b + bv + bt, s + sv + st)


    n0, n1, n2, n3 = 0, 0, 0, 0
    for head, body, stance in data:
        if stance == 0:
            if n0 >= 250:
                n0 += 1
            else:
                n0 += 1
        elif stance == 1:
            if n1 >= 250:
                n1 += 1
            else:
                n1 += 1
        elif stance == 2:
            if n2 >= 250:
                n2 += 1
            else:
                n2 += 1
        elif stance == 3:
            if n3 >= 250:
                n3 += 1
            else:
                n3 += 1
        else:
            print 'error!'

    print 'stastistics: ', n0, n1, n2, n3


if __name__ == '__main__':
    # build_baidu()
    # build_google()
    fnc()

