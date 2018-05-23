# coding: utf-8
from torch import nn
from tools import l2norm, xavier_weight
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class EnCnShare(nn.Module):
    def __init__(self, model_options):
        super(EnCnShare, self).__init__()
        # self.linear = torch.nn.Linear(model_options['dim_image'], model_options['dim'])
        self.lstm = nn.LSTM(model_options['dim_word'], model_options['dim'], 1, bidirectional=True)
        self.embedding = nn.Embedding(model_options['n_words'], model_options['dim_word'])
        self.model_options = model_options
        # self.init_weights()

    def init_weights(self):
        xavier_weight(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, en, cn):
        en_embed = self.embedding(en)
        cn = self.linear(cn)

        _, (en_embed, _) = self.lstm(en_embed)
        en_embed = en_embed.squeeze(0)

        return l2norm(en_embed), l2norm(cn)

    def forward_sens(self, x):
        x_emb = self.embedding(x)

        _, (x_emb, _) = self.lstm(x_emb)
        x_cat = x_emb.squeeze(0)
        return l2norm(x_cat)

    # def forward_imgs(self, im):
    #     im = self.linear(im)
    #     return l2norm(im)


class LIUMCVC_Encoder(nn.Module):
    def __init__(self, model_options, n_layers=1, dropout_rnn=0.5, dropout_emb=0.5,
                 dropout_ctx=0.5):
        # Initialize the Super Class
        super(LIUMCVC_Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = model_options['dim']
        self.n_direction = 2
        # Including the new dropout
        self.dropout_rnn = dropout_rnn
        self.dropout_emb = dropout_emb
        self.dropout_ctx = dropout_ctx
        # Define the Embedding Layer
        self.embedding = nn.Embedding(model_options['n_words'], model_options['dim_word'], padding_idx=0)

        # Define a Embedding Dropout_Layer, added from LIUMVIC Structure
        if dropout_emb > 0:
            self.embedding_dropout = nn.Dropout(self.dropout_emb)

        # Define a source annotation dropou_layey, added from LIUMVIC Structure
        if dropout_ctx > 0:
            self.context_dropout = nn.Dropout(self.dropout_ctx)

        # Define the LSTM Cells
        self.gru = nn.GRU(model_options['dim_word'], model_options['dim'], num_layers=n_layers, bidirectional=True,
                          dropout=self.dropout_rnn)

        self.decoderini = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.init_weights()

    def init_weights(self):
        xavier_weight(self.decoderini.weight)
        self.decoderini.bias.data.fill_(0)

    def forward(self, en, en_lengths, en_index, cn, cn_lengths, cn_index):
        """
        Input Variable:
            input_var: A variables whose size is (B,W), B is the batch size and W is the longest sequence length in the batch
            input_lengths: The lengths of each element in the batch.
            hidden: The hidden state variable whose size is (num_layer*num_directions,batch_size,hidden_size)
        Output:
            output: A variable with tensor size W*B*N, W is the maximum length of the batch, B is the batch size, and N is the hidden size
            hidden: The hidden state variable with tensor size (num_layer*num_direction,B,N)
        """
        en = self.sorted_forward(en, en_lengths, en_index)
        cn = self.sorted_forward(cn, cn_lengths, cn_index)

        return l2norm(en), l2norm(cn)

    def forward_sens(self, x, x_lengths):
        x_cat = self.single_forward(x, x_lengths)
        return l2norm(x_cat)

    def sorted_forward(self, input_var, input_lengths, index):
        # Get the mask for input_var
        ctx_mask = (input_var != 0).long().transpose(0, 1)

        # Convert input sequence into a pack_padded tensor
        embedded_x = self.embedding(input_var).transpose(0,
                                                         1)  # The dimension of embedded_x is  W*B*N, where N is the embedding size.
        if self.dropout_emb > 0:
            embedded_x = self.embedding_dropout(embedded_x)

        # Get a pack_padded sequence
        embedded_x = nn.utils.rnn.pack_padded_sequence(embedded_x, input_lengths)

        # Get an output pack_padded sequence
        output, hidden = self.gru(embedded_x)
        # Unpack the pack_padded sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output)  # The size of output will be W*B*N

        # Apply the dropout
        if self.dropout_ctx > 0:
            output = self.context_dropout(output)

        # average hidden states
        output = F.tanh(self.decoderini(output.sum(0) / ctx_mask.float().sum(0).unsqueeze(1))).unsqueeze(0)

        # reorder based on the index

        return output[0][index,]  # , ctx_mask.float()

    def unsorted_forward(self, input_var):
        # Get the mask for input_var
        ctx_mask = (input_var != 0).long().transpose(0, 1)

        # Convert input sequence into a pack_padded tensor
        embedded_x = self.embedding(input_var).transpose(0,
                                                         1)  # The dimension of embedded_x is  W*B*N, where N is the embedding size.
        if self.dropout_emb > 0:
            embedded_x = self.embedding_dropout(embedded_x)

        # Get an output pack_padded sequence
        output, hidden = self.gru(embedded_x)
        # Unpack the pack_padded sequence
        # output, _ = nn.utils.rnn.pad_packed_sequence(output)  # The size of output will be W*B*N

        # Apply the dropout
        if self.dropout_ctx > 0:
            output = self.context_dropout(output)

        # average hidden states
        output = F.tanh(self.decoderini(output.sum(0) / ctx_mask.float().sum(0).unsqueeze(1))).unsqueeze(0)

        return output[0]  # , ctx_mask.float()


class PairwiseRankingLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, cn, en):
        margin = self.margin
        # compute cn-en score matrix
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

        return cost_s.sum() + cost_im.sum()


# Agreement Classification model
class StanceClf(nn.Module):
    def __init__(self, hidden_dim, output_size):
        super(StanceClf, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_size)
        self.init_weights()

    def forward(self, batch_x, batch_y):
        x = self.dropout(torch.cat([batch_x, batch_y], dim=1))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.log_softmax(x, dim=1)
        return x

    def init_weights(self):
        xavier_weight(self.linear1.weight)
        self.linear1.bias.data.fill_(0)
        xavier_weight(self.linear2.weight)
        self.linear2.bias.data.fill_(0)
        xavier_weight(self.linear3.weight)
        self.linear3.bias.data.fill_(0)


# Simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(p=0.6)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.init_weights()

    def forward(self, x):
        x = F.relu(self.norm(self.linear1(x)))
        x = F.relu(self.norm(self.linear2(x)))
        x = F.relu(self.linear3(x))
        x = F.log_softmax(x, dim=1)
        return x

    def init_weights(self):
        xavier_weight(self.linear1.weight)
        self.linear1.bias.data.fill_(0)
        xavier_weight(self.linear2.weight)
        self.linear2.bias.data.fill_(0)
        xavier_weight(self.linear3.weight)
        self.linear3.bias.data.fill_(0)
