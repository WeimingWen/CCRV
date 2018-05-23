from models import LIUMCVC_Encoder
import torch
import cPickle as pkl
import nltk
import pynlpir
import time
import numpy as np

# run = 'final'
# data = 'UM_Corpus'
# saveto = 'save/%s' % (data)
# with open('%s_params_%s.pkl' % (saveto, run), 'rb') as f:
#     model_options = pkl.load(f)
# with open('%s.dictionary_%s.pkl' % (saveto, run), 'rb') as f:
#     worddict = pkl.load(f)
# model = LIUMCVC_Encoder(model_options)
# model.load_state_dict(torch.load('%s_model_%s.pkl' % (saveto, run)))
# model = model.cuda()

r_time = time.time()
nltk.word_tokenize(u'there is an #nisadh in the sky!!!!!')
print time.time()-r_time


r_time = time.time()
pynlpir.open()
print pynlpir.segment(u'there is an #nisadh in the sky!!!!!', pos_tagging=False)
print time.time()-r_time