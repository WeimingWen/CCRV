# coding: utf-8

import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from prepare_UM_Corpus import load_dataset
import math
from sklearn import preprocessing
import json


def encode_sentences(curr_model, pair, batch_size=128, test=True):
    """
    Encode sentences into the joint embedding space
    """
    en_feats = np.zeros((len(pair[0]), curr_model['options']['dim']), dtype='float32')
    cn_feats = np.zeros((len(pair[0]), curr_model['options']['dim']), dtype='float32')

    data_index = prepare_data(pair, curr_model['worddict'], n_words=curr_model['options']['n_words'], test=test)
    cur = 0
    for en, cn, en_lengths, cn_lengths, en_index, cn_index in data_generator(data_index, batch_size):
        en, cn = curr_model['en_cn_model'].forward(en, en_lengths, en_index, cn, cn_lengths, cn_index)
        en = en.data.cpu().numpy()
        cn = cn.data.cpu().numpy()
        for i in xrange(batch_size):
            if i + cur >= len(pair[0]):
                break
            for j in xrange(curr_model['options']['dim']):
                en_feats[i + cur][j] = en[i][j]
                cn_feats[i + cur][j] = cn[i][j]
        cur += batch_size
    en_feats = Variable(torch.from_numpy(en_feats).cuda())
    cn_feats = Variable(torch.from_numpy(cn_feats).cuda())
    return en_feats, cn_feats


def encode_sentences_np(curr_model, pair, batch_size=128, test=True):
    """
    Encode sentences into the joint embedding space
    """
    en_feats = np.zeros((len(pair[0]), curr_model['options']['dim']), dtype='float32')
    cn_feats = np.zeros((len(pair[0]), curr_model['options']['dim']), dtype='float32')

    data_index = prepare_data(pair, curr_model['worddict'],n_words=curr_model['options']['n_words'], test=test)
    cur = 0
    for en, cn, en_lengths, cn_lengths, en_index, cn_index in data_generator(data_index, batch_size):
        en, cn = curr_model['en_cn_model'].forward(en, en_lengths, en_index, cn, cn_lengths, cn_index)
        en = en.data.cpu().numpy()
        cn = cn.data.cpu().numpy()
        for i in xrange(batch_size):
            if i + cur >= len(pair[0]):
                break
            for j in xrange(curr_model['options']['dim']):
                en_feats[i + cur][j] = en[i][j]
                cn_feats[i + cur][j] = cn[i][j]
        cur += batch_size

    return en_feats, cn_feats


def l2norm(input, p=2.0, dim=1, eps=1e-12):
    """
    Compute L2 norm, row-wise
    """
    return input / input.norm(p, dim).clamp(min=eps).unsqueeze(dim=1).expand_as(input)


def xavier_weight(tensor):
    if isinstance(tensor, Variable):
        xavier_weight(tensor.data)
        return tensor

    nin, nout = tensor.size()[0], tensor.size()[1]
    r = np.sqrt(6.) / np.sqrt(nin + nout)
    return tensor.normal_(0, r)


def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = list(wordcount.keys())
    freqs = wordcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx + 2  # 0: <eos>, 1: <unk>

    return worddict  # , wordcount

def build_own_dictionary(text):
    vocab = set([])
    with open('CCMR/CCMR_Twitter_t.txt') as f1, open('CCMR/CCMR_Google_t.txt') as f2, open(
            'CCMR/CCMR_Baidu_t.txt') as f3:
        twitter = json.load(f1)
        google = json.load(f2)
        baidu = json.load(f3)
    for elem in twitter:
        for t in elem['content']:
            vocab.add(t)
    for elem in google:
        for t in elem['title']:
            vocab.add(t)
    for elem in baidu:
        for t in elem['title']:
            vocab.add(t)

    wordcount = OrderedDict()
    for words in text:
        for w in words:
            if w not in vocab:
                continue
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = list(wordcount.keys())
    freqs = wordcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx + 2  # 0: <eos>, 1: <unk>

    return worddict  # , wordcount


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq_new = seq + [0 for i in xrange(max_length - len(seq))]
    return seq_new


def data_generator(data_pairs, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index.
        batch_size: The size of the batch
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length.
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size / batch_size)
    for i in xrange(0, data_size, batch_size):
        if i + batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i + batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i + batch_size]]
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]

        # The lengths for UM_Corpus and labels to be padded to
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])

        # Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        # Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens, x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        # Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        x_index = [0 for _ in xrange(batch_size)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]
        for i in xrange(len(x_reverse_sorted_index)):
            x_index[x_reverse_sorted_index[i]] = i

        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens, y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        y_sorted_index = list(np.argsort(batch_y_lengths))
        y_reverse_sorted_index = [y for y in reversed(y_sorted_index)]
        y_index = [0 for _ in xrange(batch_size)]
        batch_y_pad_sorted = [batch_y_pad[i] for i in y_reverse_sorted_index]
        for i in xrange(len(y_reverse_sorted_index)):
            y_index[y_reverse_sorted_index[i]] = i
        # Reorder the lengths
        # batch_y_pad_sorted = [batch_y_pad[i] for i in x_reverse_sorted_index]
        # batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index]

        # Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad_sorted).cuda()), Variable(
            torch.LongTensor(batch_y_pad_sorted).cuda())
        # batch_x_lengths, batch_y_lengths = Variable(torch.LongTensor(list(reversed(sorted(batch_x_lengths))))), Variable(
        #     torch.LongTensor(list(reversed(sorted(batch_y_lengths)))))

        batch_x_lengths, batch_y_lengths = list(reversed(sorted(batch_x_lengths))), list(
            reversed(sorted(batch_y_lengths)))

        x_index, y_index = Variable(
            torch.LongTensor(x_index).cuda()), Variable(torch.LongTensor(y_index).cuda())
        # batch_x = batch_x.cuda()
        # batch_y = batch_y.cuda()
        #
        # x_index = x_index.cuda()
        # y_index = y_index.cuda()
        # Yield the batch UM_Corpus|
        yield batch_x, batch_y, batch_x_lengths, \
              batch_y_lengths, x_index, y_index


def data_generator_simple(data_pairs, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index.
        batch_size: The size of the batch
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length.
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size / batch_size)
    for i in xrange(0, data_size, batch_size):
        if i + batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i + batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i + batch_size]]
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]

        # The lengths for UM_Corpus and labels to be padded to
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])

        # Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        # Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens, x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)

        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens, y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)

        # Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad)), Variable(
            torch.LongTensor(batch_y_pad))
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # Yield the batch UM_Corpus|
        yield batch_x, batch_y, batch_x_lengths, batch_y_lengths


def prepare_data(pair, worddict, test=True, n_words=300000):
    """
    Put UM_Corpus into format useable by the model
    """
    seqs_en = []
    seqs_cn = []
    if test:
        for cc in pair[0]:
            index = []
            for w in cc:
                if worddict.get(w) is None:
                    index.append(1)
                elif worddict[w]>n_words:
                    index.append(1)
                else:
                    index.append(worddict[w])
            seqs_en.append(index)

        for cc in pair[1]:
            index = []
            for w in cc:
                if worddict.get(w) is None:
                    index.append(1)
                elif worddict[w]>n_words:
                    index.append(1)
                else:
                    index.append(worddict[w])
            seqs_cn.append(index)
    else:
        for cc in pair[0]:
            seqs_en.append([worddict[w] if worddict[w] < n_words else 1 for w in cc])
        for cc in pair[1]:
            seqs_cn.append([worddict[w] if worddict[w] < n_words else 1 for w in cc])

    return [[s, t] for s, t in zip(seqs_en, seqs_cn)]


def evalrank(model, data, split='test'):
    """
    Evaluate a trained model on either dev or test
    """

    print ('Loading dataset')
    if split == 'dev':
        _, X = load_dataset(data)
    else:
        X = load_dataset(data, load_test=True)

    print ('Computing results...')
    en, cn = encode_sentences(model, X, test=True)

    score = devloss(en, cn, margin=model['options']['margin'])

    print(split + ' loss: ', score)
    # (r1, r5, r10, medr) = i2t(cn, en)
    # print ("cn to en: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
    # (r1i, r5i, r10i, medri) = t2i(cn, en)
    # print ("en to cn: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))


def devloss(en, cn, margin=0.2):
    scores = torch.mm(cn, en.transpose(1, 0))
    diagonal = scores.diag()

    # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
    cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
                       (margin - diagonal).expand_as(scores) + scores)
    # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
    cost_im = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
                        (margin - diagonal).expand_as(scores).transpose(1, 0) + scores)

    for i in xrange(scores.size()[0]):
        cost_s[i, i] = 0
        cost_im[i, i] = 0

    return (cost_s.sum() + cost_im.sum()).data.cpu().numpy()[0]

def devloss_cpu(en, cn, margin=0.2):
    scores = torch.mm(cn, en.transpose(1, 0))
    diagonal = scores.diag()

    # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
    cost_s = torch.max(torch.zeros(scores.size()[0], scores.size()[1]),
                       (margin - diagonal).expand_as(scores) + scores)
    # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
    cost_im = torch.max(torch.zeros(scores.size()[0], scores.size()[1]),
                        (margin - diagonal).expand_as(scores).transpose(1, 0) + scores)

    for i in xrange(scores.size()[0]):
        cost_s[i, i] = 0
        cost_im[i, i] = 0

    return (cost_s.sum() + cost_im.sum()).data[0]


def imputer(data):
    imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(data)
    data = imp.transform(data)
    return data


# scale to (0, 1)
def standardize(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    data = scaler.transform(data)
    return data


def metrics(y, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y[i] == 0 and y_pred[i] == 0:
            pass
        else:
            print 'error!'
    # if (tp + fp) == 0:
    #     p = 1
    # else:
    #     p = float(tp) / (tp + fp)
    # if (tp + fn) == 0:
    #     recall = 1
    # else:
    #     recall = float(tp) / (tp + fn)
    if (tp + fp + fn) == 0:
        F1 = 1
    else:
        F1 = 2 * float(tp) / (2 * tp + fp + fn)
    return F1
