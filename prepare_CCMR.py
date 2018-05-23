import json
from nlp import Tokenizer


def tokenize_data():
    t = Tokenizer()

    with open('CCMR/CCMR_Google.txt') as f1, open('CCMR/CCMR_Baidu.txt') as f2, open('CCMR/CCMR_Twitter.txt') as f3:
        google = json.load(f1)
        baidu = json.load(f2)
        twitter = json.load(f3)

    print 'start google ...'
    for elem in google:
        elem['title'] = t.tokenize(elem['title'])

    print 'start baidu ...'
    for elem in baidu:
        elem['title'] = t.tokenize(elem['title'], cn=True)

    print 'start twitter ...'
    for elem in twitter:
        elem['content'] = t.tokenize(elem['content'])

    print 'save ...'
    with open('CCMR/CCMR_Google_t.txt', 'w') as f1, open('CCMR/CCMR_Baidu_t.txt','w') as f2, open('CCMR/CCMR_Twitter_t.txt','w') as f3:
        json.dump(google,f1)
        json.dump(baidu,f2)
        json.dump(twitter,f3)


if __name__ == '__main__':
    tokenize_data()