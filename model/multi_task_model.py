import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, npostag, ninp, npostagemb, nhid, postagnhid, nlayers, postagnlayers, dropout=0.5, postag_dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.postag_encoder = nn.Embedding(npostag, npostagemb)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp + npostagemb, nhid, nlayers, dropout=dropout)
            self.postag_rnn = getattr(nn, rnn_type)(npostagemb, postagnhid, postagnlayers, dropout=postag_dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)
        self.postag_decoder = nn.Linear(postagnhid, npostag)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for postaguage Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

            if postagnhid != npostagemb:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.postag_decoder.weight = self.postag_encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.postag_encoder.weight.data.uniform_(-initrange, initrange)
        
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

        self.postag_decoder.bias.data.fill_(0)
        self.postag_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, input_postag, hidden, hidden_postag):
        emb = self.drop(self.encoder(input))
        postag_emb = self.drop(self.postag_encoder(input_postag))

        merge_emb = torch.cat((emb, postag_emb), dim=2)

        output, hidden = self.rnn(merge_emb, hidden)
        output = self.drop(output)
        output_postag, hidden_postag = self.postag_rnn(postag_emb, hidden_postag)
        output_postag = self.drop(output_postag)

        new_output = output + output_postag

        decoded = F.log_softmax(self.decoder(new_output.view(new_output.size(0)*new_output.size(1), new_output.size(2))))
        postag_decoded = F.log_softmax(self.postag_decoder(output_postag.view(output_postag.size(0)*output_postag.size(1), output_postag.size(2))))
        return decoded.view(new_output.size(0), new_output.size(1), decoded.size(1)), postag_decoded.view(output_postag.size(0), output_postag.size(1), postag_decoded.size(1)), hidden, hidden_postag

    def init_hidden(self, layer, bsz, nhid):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(layer, bsz, nhid).zero_()),
                    Variable(weight.new(layer, bsz, nhid).zero_()))
        else:
            return Variable(weight.new(layer, bsz, nhid).zero_())