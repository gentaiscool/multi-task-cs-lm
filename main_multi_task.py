import argparse
import time
import math
import os
import unicodedata

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import util.data_multi_task as data
import model.multi_task_model as model

import util.printhelper as printhelper

parser = argparse.ArgumentParser(description='PyTorch SEAME RNN/LSTM Language Model')
parser.add_argument('--name', type=str, default='',
                    help='name')
parser.add_argument('--data', type=str, default='../data/seame_phase2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--postagemsize', type=int, default=200,
                    help='size of postag embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--postagnhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--postagnlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to lm rnn layers (0 = no dropout)')
parser.add_argument('--postagdropout', type=float, default=0.2,
                    help='dropout applied to postag rnn layers (0 = no dropout)')
parser.add_argument('--p', type=float, default=1.0,
                    help='ratio loss lm')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_path', type=str, default='./log', help='location of log file')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./model',
                    help='path to save the final model')
args = parser.parse_args()

alpha = float(args.p)
beta = 1.0 - float(args.p)

log_name = str(args.name) + "_model" + str(args.model) + "_alpha" + str(args.alpha) + "_beta" + str(args.beta) + "_clip" + str(args.clip) + "_layers" + str(args.nlayers) + "_nhid" + str(args.nhid) + "_emsize" + str(args.emsize) + "_postagemsize" + str(args.postagemsize) + ".txt"
log_file = open(args.log_path + "/" + log_name, "w+")

save_path = args.save + "/" + log_name + ".pt"
dir_path = os.path.dirname(os.path.realpath(__file__))

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Write all summary
printhelper.print_log(log_file, "alpha\t:" + str(args.alpha))
printhelper.print_log(log_file, "beta\t:" + str(args.beta))
printhelper.print_log(log_file, "clip\t:" + str(args.clip))
printhelper.print_log(log_file, "data\t:" + str(args.data))
printhelper.print_log(log_file, "start lr\t:" + str(args.lr))
printhelper.print_log(log_file, "em size\t:" + str(args.emsize))
printhelper.print_log(log_file, "postag em size\t:" + str(args.postagemsize))

def is_chinese_char(cc):
    return unicodedata.category(cc) == 'Lo'

def is_contain_chinese_word(seq):
    for i in range(len(seq)):
        if is_chinese_char(seq[i]):
            return True
    return False

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

train_postag_data = batchify(corpus.train_postag, args.batch_size)
val_postag_data = batchify(corpus.valid_postag, eval_batch_size)
test_postag_data = batchify(corpus.test_postag, eval_batch_size)

# per word_sequence
val_sequence_data = corpus.valid_seq_idx_matrix
test_sequence_data = corpus.test_seq_idx_matrix

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
npostag = corpus.dictionary.get_postag_len()
model = model.RNNModel(args.model, ntokens, npostag, args.emsize, args.postagemsize, args.nhid, args.postagnhid, args.nlayers, args.postagnlayers, args.dropout, args.postagdropout, args.tied)

printhelper.print_log(log_file, str(model))
if args.cuda:
    model.cuda()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word
pos2idx = corpus.dictionary.pos2idx
idx2pos = corpus.dictionary.idx2pos

num_word = len(corpus.dictionary.idx2word)
english_word = {}
chinese_word = {}
for j in range(num_word):
    word = idx2word[j]
    if is_contain_chinese_word(word):
        chinese_word[j] = True
    else:
        english_word[j] = True

def evaluate(data_source, data_source_postag, type_evaluation="val"):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.nlayers, eval_batch_size, args.nhid)
    hidden_postag = model.init_hidden(args.postagnlayers, eval_batch_size, args.postagnhid)
    criterion = nn.CrossEntropyLoss()

    if type_evaluation == "test":
        file_out = open("predictions/" + log_name, "w+", encoding="utf-8")

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        data_postag, targets_postag = get_batch(data_source_postag, i, evaluation=True)

        output, postag_output, hidden, hidden_postag = model(data, data_postag, hidden, hidden_postag) # data (batch_size, seq_len), postag_output (batch_size, seq_len, labels)
        num_postag = corpus.dictionary.get_postag_len()

        english_postag = {}
        chinese_postag = {}
        for j in range(num_postag):
            if idx2pos[j] == '<oop>':
                english_postag[j] = True
                continue

            postag_lang = idx2pos[j].split("_")[1]
            if postag_lang == "EN":
                english_postag[j] = True
            else:
                chinese_postag[j] = True

        if type_evaluation == "test":
            batch_size = postag_output.size(1)
            seq_len = postag_output.size(0)
            for k in range(batch_size):
                for j in range(seq_len-1):
                    en_val = 0
                    zh_val = 0

                    en_word_val = 0
                    zh_word_val = 0

                    postag = postag_output[j][k]
                    for l in range(len(postag)):
                        if l in english_postag:
                            en_val += pow(math.e, postag[l].data[0])
                        else:
                            zh_val += pow(math.e, postag[l].data[0])

                    word_dist = output[j][k]
                
                    target_batch = targets.view(seq_len, batch_size)
                    target_word = target_batch[j][k].data[0]
                    word_val = pow(math.e, word_dist[target_word].data[0])
                    word_val_log = word_dist[target_word].data[0]

                    file_out.write(idx2word[data[j][k].data[0]] + "\t" + idx2word[data[j+1][k].data[0]] + "\t" + str(word_val) + "\t" + str(word_val_log) + "\t" + str(en_val) + "\t" + str(zh_val) + "\n")
                file_out.write("\n")

        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        hidden_postag = repackage_hidden(hidden_postag)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.nlayers, args.batch_size, args.nhid)
    hidden_postag = model.init_hidden(args.postagnlayers, args.batch_size, args.postagnhid)
    criterion = nn.CrossEntropyLoss()
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data_postag, targets_postag = get_batch(train_postag_data, i)
        data, targets = get_batch(train_data, i)
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        hidden_postag = repackage_hidden(hidden_postag)
        model.zero_grad()
        
        output, postag_output, hidden, hidden_postag = model(data, data_postag, hidden, hidden_postag)

        postag_loss = criterion(postag_output.view(-1, npostag), targets_postag)
        loss = criterion(output.view(-1, ntokens), targets)

        accu_loss = alpha * loss + beta * postag_loss
        accu_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        opt = optim.SGD(model.parameters(), lr=lr)
        opt.step()

        total_loss += loss.data
        
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time

            log = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | word_loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss))

            printhelper.print_log(log_file, log)

            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

counter = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data, val_postag_data, "val")

        log = '-' * 89 + "\n" + '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)) + '-' * 89
        printhelper.print_log(log_file, log)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save_path, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            counter = 0
        else:
            lr /= 4.0
            counter += 1

        if counter == 5:
            break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(save_path, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data, test_postag_data, "test")

log = ('=' * 89) + '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)) + ('=' * 89)
printhelper.print_log(log_file, log)
log_file.close()