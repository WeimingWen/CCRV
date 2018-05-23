# -*- coding: utf-8 -*-

import nltk
import re
import pynlpir
import time

# tokenize texts with pynlpir
class Tokenizer():
    def __init__(self):
        pynlpir.open()

    def tokenize(self, text, cn=False):
        try:
            text = unicode(text, encoding='utf-8', errors='replace')
        except:
            pass
        text = text.lower()
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'URL', text)
        text = re.sub('(\@[^\s]+)', 'USER', text)

        if cn:
            try:
                return pynlpir.segment(text, pos_tagging=False)
            except:
                # print 'pynlpir cannot tokenize: ', text
                return nltk.word_tokenize(text)
        else:
            return nltk.word_tokenize(text)

    def close(self):
        pynlpir.close()
