import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score
from models import MLP
from torch import optim
import numpy as np
import os


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .CLCPFeatures
    category_i = top_i[:, 0].cpu().numpy()  # print(stance.size(0))
    return category_i


def train_MLP(train_X, train_Y, test_X, test_Y, batch_size=20, epochs=100):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    model = MLP(10, 20, 2)
    model.cuda()
    model.train()

    learn_rate = 1e-3
    grad_clip = 2.0
    dispFreq = 50
    validFreq = 200
    early_stop = 20
    weight=torch.FloatTensor([2.0,1.0])
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    params = filter(lambda p: p.requires_grad, model.parameters())

    dev_tensor = Variable(torch.FloatTensor(test_X).cuda())

    curr = 0
    uidx = 0
    # For Early-stopping

    best_step = 0
    for iepx in xrange(1, epochs + 1):
        for ibx in xrange(0, len(train_X), batch_size):
            if ibx + batch_size >= len(train_X):
                batch = Variable(torch.FloatTensor(train_X[ibx:len(train_X)]).cuda())
                target = Variable(torch.LongTensor(train_Y[ibx:len(train_X)]).cuda())
            else:
                batch = Variable(torch.FloatTensor(train_X[ibx:ibx + batch_size]).cuda())
                target = Variable(torch.LongTensor(train_Y[ibx:ibx + batch_size]).cuda())

            uidx += 1

            pred = model(batch)

            loss = loss_function(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, grad_clip)
            optimizer.step()

            if np.mod(uidx, dispFreq) == 0:
                print 'Epoch ', iepx, '\tUpdate ', uidx, '\tCost ', loss.data.cpu().numpy()[0]

            if np.mod(uidx, validFreq) == 0:
                # compute dev
                model.eval()
                out = model.forward(dev_tensor)
                model.train()
                # score = nn.NLLLoss(weight=weight)(out, vs_tensor).data[0]
                pred = categoryFromOutput(out)
                F1 = f1_score(test_Y, pred)

                curr_step = uidx / validFreq

                currscore = F1

                print 'F1 on dev', F1

                if currscore > curr:
                    curr = currscore
                    best_step = curr_step

                    # Save model
                    print 'Saving model...',
                    # torch.save(model.state_dict(), '%s_model_%s.pkl' % (saveto, run))
                    print 'Done'

                if curr_step - best_step > early_stop:
                    print 'Early stopping ...'
                    print best_step
                    print curr
                    return


