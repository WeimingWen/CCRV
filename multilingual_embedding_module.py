import torch
import cPickle as pkl
from tools import encode_sentences_np
from models import LIUMCVC_Encoder


# Multilingual Embedding Module
class Embedder():
    def __init__(self, run):
        data = 'UM_Corpus'
        saveto = 'save/%s' % (data)
        with open('%s_params_%s.pkl' % (saveto, run), 'rb') as f:
            model_options = pkl.load(f)
        with open('%s.dictionary_%s.pkl' % (saveto, run), 'rb') as f:
            worddict = pkl.load(f)
        model = LIUMCVC_Encoder(model_options)
        model.load_state_dict(torch.load('%s_model_%s.pkl' % (saveto, run)))
        model = model.cuda()
        model.eval()

        best_model = {}
        best_model['options'] = model_options
        best_model['en_cn_model'] = model
        best_model['worddict'] = worddict
        self.model = best_model

    def embed(self, text1, text2):
        feats1, feats2 = encode_sentences_np(self.model, (text1, text2), test=True)
        return feats1, feats2