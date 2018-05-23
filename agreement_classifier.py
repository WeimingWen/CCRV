from models import StanceClf
import torch
import numpy as np
from torch.autograd import Variable


# Agreement Classifier
class Stancer:
    def __init__(self, run):
        data = 'FNC_1'
        path = 'save/%s' % (data)
        model = StanceClf(600, 4)
        model.load_state_dict(torch.load('%s_model_%s.pkl' % (path, run)))
        model = model.cuda()
        model.eval()
        self.model = model

    def compute_stance(self, head, body):
        head = Variable(torch.FloatTensor(head).cuda())
        body = Variable(torch.FloatTensor(body).cuda())
        out = self.model.forward(head, body)
        return np.exp(out.data.cpu().numpy())
