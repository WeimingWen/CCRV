import numpy as np
import cPickle as pkl
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score
from models import StanceClf
from multilingual_embedding_module import Embedder

stances = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .CLCPFeatures
    category_i = top_i[:, 0].cpu().numpy()  # print(stance.size(0))
    return category_i


def train():
    run = 'complete'
    data = 'FNC_1'
    saveto = 'save/%s' % (data)

    output_size = 4
    sent_dim = 300
    hidden_dim = sent_dim * 2
    batch_size = 32
    learn_rate = 1e-5
    grad_clip = 2.0
    max_epoch = 100
    validFreq = 200
    dispFreq = 50
    early_stop = 50

    model = StanceClf(hidden_dim, output_size)
    model = model.cuda()
    model.train()

    # class weight: agree disagree discuss unrelated
    weight = Variable(torch.FloatTensor([54644.0/5331.0, 54644.0/1287.0, 54644.0/13123.0, 1.0])).cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    loss_function = nn.NLLLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    with open('FNC_1/train.pkl', 'rb') as f:
        train_tensor = pkl.load(f)

    with open('FNC_1/dev.pkl', 'rb') as f:
        vh, vb, vs = pkl.load(f)
    vh = Variable(vh.cuda())
    vb = Variable(vb.cuda())
    #vs_tensor = Variable(torch.LongTensor(vs).cuda())

    print 'start training ...'
    curr = 0
    uidx = 0
    # For Early-stopping
    best_step = 0
    for iepx in xrange(1, max_epoch + 1):
        for head, body, stance in train_tensor:
            head = Variable(torch.FloatTensor(head).cuda())
            body = Variable(torch.FloatTensor(body).cuda())
            stance = Variable(torch.LongTensor(stance).cuda())

            uidx += 1
            pred = model(head, body)

            loss = loss_function(pred, stance)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, grad_clip)
            optimizer.step()

            if np.mod(uidx, dispFreq) == 0:
                print 'Epoch ', iepx, '\tUpdate ', uidx, '\tCost ', loss.data.cpu().numpy()[0]

            if np.mod(uidx, validFreq) == 0:
                # compute dev
                model.eval()
                out = model.forward(vh, vb)
                model.train()
                #score = nn.NLLLoss(weight=weight)(out, vs_tensor).data[0]
                pred = categoryFromOutput(out)
                acc = accuracy_score(vs, pred)
                F1 = f1_score(vs, pred, average='macro')

                curr_step = uidx / validFreq

                currscore = F1

                print 'acc on dev', acc
                print 'F1 on dev', F1

                if currscore > curr:
                    curr = currscore
                    best_step = curr_step

                    # Save model
                    print 'Saving model...',
                    torch.save(model.state_dict(), '%s_model_%s.pkl' % (saveto, run))
                    print 'Done'

                if curr_step - best_step > early_stop:
                    print 'Early stopping ...'
                    return


if __name__ == '__main__':
    train()
